//! The checks over `commands.toml` and `divergences.toml`.
//!
//! `12` section 3 states the gate in one sentence: a command with no storage
//! plan does not ship. `12` section 10 states the other one: a new divergence
//! requires an entry in the register in the same commit that creates it, and CI
//! checks that every command marked divergent has a row.
//!
//! Both of those are only worth writing down if something enforces them, which
//! is what this is. It runs inside `cargo xtask check` next to the generated
//! file diff, so a pull request that adds a command without a plan or a
//! divergence without a row fails the same job that a stale header fails.

use crate::toml::{self, Table};
use std::collections::BTreeSet;
use std::fs;

/// The storage plans a command may claim (`12` section 3).
const PLANS: &[&str] = &[
    "point",
    "cursor",
    "merge",
    "metadata",
    "whole-value",
    "none",
];
/// The groups whose commands are allowed to claim the `none` plan.
///
/// A command that touches no key has no storage plan to state, and there are
/// three groups that can honestly say so. Anywhere else it is a command whose
/// plan nobody has worked out, which is the thing `12` section 3 exists to
/// stop.
///
/// Scripting is on the list for the two container commands and not for the six
/// that run code. SCRIPT and FUNCTION are about a cache and a library set and
/// never name a key. EVAL and the rest take their keys in an argument and are
/// in the `merge` plan when they land, so adding the group here does not let
/// them through without one.
const PLANLESS_GROUPS: &[&str] = &["connection", "scripting", "server"];
/// The commands allowed to claim the `none` plan outside those groups.
///
/// Two of them, and they are named one by one rather than by widening the group
/// list, because the keyspace group is where the commands that do touch keys
/// live and letting it through wholesale would let `DEL` ship without a plan.
///
/// `WAIT` and `WAITAOF` are in Redis's generic group, which is this group, and
/// they name no key and read nothing. They ask about replication and about the
/// append only file, which are both facts about the server rather than about a
/// value, and Redis puts them in the `@connection` ACL category for exactly that
/// reason. Their group is where it is so that the count in `commands.toml`
/// matches what a real server answers, and the honest plan for them is `none`.
/// `PFSELFTEST` is in the HyperLogLog group because that is the group a real
/// server puts it in, and it names no key and reads none. It asks whether the
/// sketch code works, which is a fact about the build rather than about a value,
/// and the honest plan for it is `none` as well.
/// The sixteen search commands that are about an index rather than a document
/// are here for the same reason and are named one by one for the same reason.
/// An index is not a key, none of the sixteen names one, and `COMMAND INFO`
/// reports no key spec for any of them. The rest of the search group is a
/// different matter: `FT.SEARCH` and `FT.AGGREGATE` read documents out of the
/// keyspace and have a real plan to state, so widening the group here would let
/// the commands the gate is for through without one.
const PLANLESS_COMMANDS: &[&str] = &[
    "WAIT",
    "WAITAOF",
    "PFSELFTEST",
    "FT.CREATE",
    "FT._CREATEIFNX",
    "FT.ALTER",
    "FT._ALTERIFNX",
    "FT.DROPINDEX",
    "FT._DROPINDEXIFX",
    "FT.DROP",
    "FT._DROPIFX",
    "FT.INFO",
    "FT._LIST",
    "FT.ALIASADD",
    "FT._ALIASADDIFNX",
    "FT.ALIASDEL",
    "FT._ALIASDELIFX",
    "FT.ALIASUPDATE",
    "FT.ALIASLIST",
];
/// The bound or materialise verdicts.
const BOUNDED: &[&str] = &["inherent", "yes", "risk"];
/// Whether a command is implemented, and how far up it reaches.
const STATUSES: &[&str] = &["shipped", "planned"];
/// The same for a group, which can also be halfway through.
const GROUP_STATUSES: &[&str] = &["shipped", "partial", "planned"];
/// How far the argument order has been checked.
const WIRE: &[&str] = &["verified", "unverified", "none"];
/// What a divergence row promises.
const DIVERGENCE_STATUSES: &[&str] = &["permanent", "until"];

/// Everything wrong with the two registry files, in the order it was found.
pub fn problems() -> Vec<String> {
    let root = crate::root();
    let mut bad = Vec::new();

    let commands = match read(&root.join("commands.toml")) {
        Ok(t) => t,
        Err(e) => return vec![e],
    };
    let divergences = match read(&root.join("divergences.toml")) {
        Ok(t) => t,
        Err(e) => return vec![e],
    };

    let ids = check_divergences(&divergences, &mut bad);
    check_commands(&commands, &ids, &mut bad);
    check_table(&commands, &mut bad);
    bad
}

/// The dispatch table and the audit have to say the same thing about the same
/// command.
///
/// Two files describing one command is two files that will disagree, and the
/// interesting direction is not the obvious one. A command missing from
/// `commands.toml` is a command that shipped without a storage plan, which is
/// the gate from `12` section 3. A row claiming `wire = "verified"` for a
/// command nothing dispatches is a claim about an argument order that no code
/// has, which is worse than an unverified row because it reads as done.
fn check_table(tables: &[Table], bad: &mut Vec<String>) {
    let rows: Vec<&Table> = tables.iter().filter(|t| t.name == "command").collect();
    let row = |name: &str| {
        let upper = name.to_uppercase();
        rows.iter()
            .copied()
            .find(|t| t.str("name").unwrap_or_default() == upper)
    };

    for spec in yo_resp::dispatch::COMMANDS {
        let Some(t) = row(spec.name) else {
            bad.push(format!(
                "commands.toml: {} is dispatched and is not listed",
                spec.name.to_uppercase()
            ));
            continue;
        };
        let name = spec.name.to_uppercase();
        let arity = t.get("arity").and_then(|v| v.as_int()).unwrap_or_default();
        if arity != i64::from(spec.arity) {
            bad.push(format!(
                "commands.toml line {}: {name} has arity {arity} and the dispatch table has {}",
                t.line, spec.arity
            ));
        }
        let group = t.str("group").unwrap_or_default();
        if group != spec.group {
            bad.push(format!(
                "commands.toml line {}: {name} is in group {group} and the dispatch table says {}",
                t.line, spec.group
            ));
        }
        for (key, want) in [("status", "shipped"), ("wire", "verified")] {
            let got = t.str(key).unwrap_or_default();
            if got != want {
                bad.push(format!(
                    "commands.toml line {}: {name} is dispatched and has {key} = {got}",
                    t.line
                ));
            }
        }
    }

    for t in rows {
        let name = t.str("name").unwrap_or_default();
        if t.str("wire") == Ok("verified") && yo_resp::dispatch::lookup(name.as_bytes()).is_none() {
            bad.push(format!(
                "commands.toml line {}: {name} says its wire is verified and nothing dispatches it",
                t.line
            ));
        }
    }
}

fn read(path: &std::path::Path) -> Result<Vec<Table>, String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .replace("\r\n", "\n");
    toml::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Checks the register and returns the ids it defines.
fn check_divergences(tables: &[Table], bad: &mut Vec<String>) -> BTreeSet<String> {
    let mut ids = BTreeSet::new();
    for t in tables {
        if t.name != "divergence" {
            bad.push(format!(
                "divergences.toml line {}: [[{}]] is not a divergence",
                t.line, t.name
            ));
            continue;
        }
        let id = match t.str("id") {
            Ok(s) => s.to_string(),
            Err(e) => {
                bad.push(format!("divergences.toml {e}"));
                continue;
            }
        };
        for key in ["title", "reason", "spec"] {
            if let Err(e) = t.str(key) {
                bad.push(format!("divergences.toml {e}"));
            }
        }
        match t.str("status") {
            Ok(s) if DIVERGENCE_STATUSES.contains(&s) => {
                // A divergence that goes away has to say when, or it is a
                // permanent one that nobody has admitted to yet.
                if s == "until" && t.str("resolved_by").is_err() {
                    bad.push(format!(
                        "divergences.toml line {}: {id} is temporary and does not say which milestone resolves it",
                        t.line
                    ));
                }
                if s == "permanent" && t.get("resolved_by").is_some() {
                    bad.push(format!(
                        "divergences.toml line {}: {id} is permanent and names a milestone that resolves it",
                        t.line
                    ));
                }
            }
            Ok(s) => bad.push(format!(
                "divergences.toml line {}: {id} has status {s}, which is not one of {DIVERGENCE_STATUSES:?}",
                t.line
            )),
            Err(e) => bad.push(format!("divergences.toml {e}")),
        }
        if !ids.insert(id.clone()) {
            bad.push(format!(
                "divergences.toml line {}: {id} is defined twice",
                t.line
            ));
        }
    }
    ids
}

fn check_commands(tables: &[Table], ids: &BTreeSet<String>, bad: &mut Vec<String>) {
    let mut names = BTreeSet::new();
    let mut groups = Vec::new();
    let mut counted: Vec<(String, String, bool)> = Vec::new();

    for t in tables {
        match t.name.as_str() {
            "group" => groups.push(t),
            "command" => {}
            other => {
                bad.push(format!(
                    "commands.toml line {}: [[{other}]] is neither a group nor a command",
                    t.line
                ));
                continue;
            }
        }
        if t.name == "group" {
            continue;
        }

        let name = match t.str("name") {
            Ok(s) => s.to_string(),
            Err(e) => {
                bad.push(format!("commands.toml {e}"));
                continue;
            }
        };
        if !names.insert(name.clone()) {
            bad.push(format!(
                "commands.toml line {}: {name} is listed twice",
                t.line
            ));
        }
        if name != name.to_uppercase() {
            bad.push(format!(
                "commands.toml line {}: {name} is not upper case, and the wire is",
                t.line
            ));
        }

        let group = t.str("group").unwrap_or_default().to_string();
        if group.is_empty() {
            bad.push(format!(
                "commands.toml line {}: {name} has no group",
                t.line
            ));
        }
        let plan = one_of(t, "plan", PLANS, &name, bad);
        one_of(t, "bounded", BOUNDED, &name, bad);
        one_of(t, "wire", WIRE, &name, bad);
        let status = one_of(t, "status", STATUSES, &name, bad);
        counted.push((
            group.clone(),
            name.clone(),
            status.as_deref() == Some("shipped"),
        ));

        if plan.as_deref() == Some("none")
            && !PLANLESS_GROUPS.contains(&group.as_str())
            && !PLANLESS_COMMANDS.contains(&name.as_str())
        {
            bad.push(format!(
                "commands.toml line {}: {name} claims no storage plan and is in group {group}, which is not one of {PLANLESS_GROUPS:?}",
                t.line
            ));
        }

        // The gate from `12` section 3, in the only form that means anything.
        if status.as_deref() == Some("shipped") && t.get("plan").is_none() {
            bad.push(format!(
                "commands.toml line {}: {name} is shipped and has no storage plan",
                t.line
            ));
        }
        if t.get("since").is_none() {
            bad.push(format!(
                "commands.toml line {}: {name} does not say which Redis it came from",
                t.line
            ));
        }
        if t.get("arity").and_then(|v| v.as_int()).is_none() {
            bad.push(format!(
                "commands.toml line {}: {name} has no arity, and COMMAND has to report one",
                t.line
            ));
        }

        // `12` section 10, the other half of the register.
        // One id, or several separated by commas. A command can diverge twice
        // for two unrelated reasons with two different lifetimes, which XINFO
        // does: one of its two goes away when a feature lands and the other is
        // permanent, so they cannot be written as one row.
        if let Some(v) = t.get("divergent") {
            match v.as_str() {
                Some(list) => {
                    for id in list.split(',').map(str::trim) {
                        if !ids.contains(id) {
                            bad.push(format!(
                                "commands.toml line {}: {name} names divergence {id}, which is not in divergences.toml",
                                t.line
                            ));
                        }
                    }
                }
                None => bad.push(format!(
                    "commands.toml line {}: {name} has divergent = {v}, which should be a divergence id",
                    t.line
                )),
            }
        }
    }

    check_group_budgets(&groups, &counted, bad);
}

/// A group that says it is shipped has to hold the number of commands it claims.
///
/// This is what stops the count in `12` section 3 and the file drifting apart
/// silently. When they disagree, one of the two is wrong and somebody has to
/// say which, which is exactly what happened to the string group's 28.
///
/// What is counted is the commands that say they are shipped, not the rows in
/// the file. Those were the same thing until Redis 8.10 added commands nobody
/// has written yet, and a row for one of those is the file being honest about
/// what the group owes rather than the group having grown. Counting rows would
/// have meant either leaving the new commands out of the file or claiming them
/// as done, and both of those are the drift this check exists to catch.
fn check_group_budgets(
    groups: &[&Table],
    counted: &[(String, String, bool)],
    bad: &mut Vec<String>,
) {
    let mut seen = BTreeSet::new();
    for g in groups {
        let name = match g.str("name") {
            Ok(s) => s.to_string(),
            Err(e) => {
                bad.push(format!("commands.toml {e}"));
                continue;
            }
        };
        if !seen.insert(name.clone()) {
            bad.push(format!(
                "commands.toml line {}: group {name} is declared twice",
                g.line
            ));
        }
        let Some(expected) = g.get("expected").and_then(|v| v.as_int()) else {
            bad.push(format!(
                "commands.toml line {}: group {name} has no expected count",
                g.line
            ));
            continue;
        };
        let status = g.str("status").unwrap_or_default();
        if !GROUP_STATUSES.contains(&status) {
            bad.push(format!(
                "commands.toml line {}: group {name} has status {status}, which is not one of {GROUP_STATUSES:?}",
                g.line
            ));
        }
        let have = counted
            .iter()
            .filter(|(gr, _, shipped)| *gr == name && *shipped)
            .count() as i64;
        if status == "shipped" && have != expected {
            bad.push(format!(
                "commands.toml line {}: group {name} says {expected} commands and lists {have}",
                g.line
            ));
        }
        if status == "planned" && have != 0 {
            bad.push(format!(
                "commands.toml line {}: group {name} is planned and already lists {have} commands, so it is partial",
                g.line
            ));
        }
        // A group being worked through has to be somewhere between the two, or
        // its status is a stale answer to a question that has moved on.
        if status == "partial" && (have == 0 || have >= expected) {
            bad.push(format!(
                "commands.toml line {}: group {name} is partial and lists {have} of {expected}",
                g.line
            ));
        }
    }
    for (group, name, _) in counted {
        if !group.is_empty() && !seen.contains(group) {
            bad.push(format!(
                "commands.toml: {name} is in group {group}, which is not declared"
            ));
        }
    }
}

/// Reads a key that has to be one of a fixed set, complaining if it is not.
fn one_of(
    t: &Table,
    key: &str,
    allowed: &[&str],
    name: &str,
    bad: &mut Vec<String>,
) -> Option<String> {
    match t.str(key) {
        Ok(s) if allowed.contains(&s) => Some(s.to_string()),
        Ok(s) => {
            bad.push(format!(
                "commands.toml line {}: {name} has {key} = {s}, which is not one of {allowed:?}",
                t.line
            ));
            None
        }
        Err(e) => {
            bad.push(format!("commands.toml {e}"));
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The registry files as checked in have to pass, or the check below is
    /// only ever going to be run by somebody who has already broken something.
    #[test]
    fn the_registry_files_are_consistent() {
        let bad = problems();
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }

    /// The string group is the one that is shipped, and a real Redis 8.8 answers
    /// 26 where `12` section 3 says 28. If that ever changes, the spec and the
    /// file have to change together.
    #[test]
    fn the_string_group_is_complete() {
        let root = crate::root();
        let tables = read(&root.join("commands.toml")).unwrap();
        let strings: Vec<_> = tables
            .iter()
            .filter(|t| t.name == "command" && t.str("group").unwrap_or_default() == "string")
            .collect();
        assert_eq!(strings.len(), 26);
        for want in [
            "SET",
            "GET",
            "GETSET",
            "GETDEL",
            "GETEX",
            "SETNX",
            "SETEX",
            "PSETEX",
            "MSET",
            "MSETNX",
            "MGET",
            "APPEND",
            "STRLEN",
            "SETRANGE",
            "GETRANGE",
            "SUBSTR",
            "INCR",
            "DECR",
            "INCRBY",
            "DECRBY",
            "INCRBYFLOAT",
            "LCS",
            "MSETEX",
            "DELEX",
            "DIGEST",
            "INCREX",
        ] {
            assert!(
                strings.iter().any(|t| t.str("name").unwrap() == want),
                "{want} is not in commands.toml"
            );
        }
        // Everything in the group is implemented, so everything in the group
        // has a plan and a verdict.
        for t in &strings {
            assert_eq!(t.str("status").unwrap(), "shipped", "{:?}", t.str("name"));
            assert!(PLANS.contains(&t.str("plan").unwrap()));
        }
    }

    /// A command that names a divergence nobody wrote down is the failure this
    /// whole file exists to catch, so it is worth checking that it is caught.
    #[test]
    fn a_divergence_that_is_not_registered_is_caught() {
        let ids = BTreeSet::from(["D-1".to_string()]);
        let tables = toml::parse(
            "[[group]]\nname = \"string\"\nexpected = 1\nstatus = \"shipped\"\n\n\
             [[command]]\nname = \"SET\"\ngroup = \"string\"\nsince = \"1.0.0\"\narity = -3\n\
             plan = \"point\"\nbounded = \"inherent\"\nstatus = \"shipped\"\nwire = \"none\"\n\
             divergent = \"D-99\"\n",
        )
        .unwrap();
        let mut bad = Vec::new();
        check_commands(&tables, &ids, &mut bad);
        assert_eq!(bad.len(), 1, "{bad:?}");
        assert!(bad[0].contains("D-99"), "{}", bad[0]);
    }

    #[test]
    fn a_shipped_command_with_no_plan_is_caught() {
        let tables = toml::parse(
            "[[group]]\nname = \"string\"\nexpected = 1\nstatus = \"shipped\"\n\n\
             [[command]]\nname = \"SET\"\ngroup = \"string\"\nsince = \"1.0.0\"\narity = -3\n\
             bounded = \"inherent\"\nstatus = \"shipped\"\nwire = \"none\"\n",
        )
        .unwrap();
        let mut bad = Vec::new();
        check_commands(&tables, &BTreeSet::new(), &mut bad);
        assert!(bad.iter().any(|b| b.contains("no storage plan")), "{bad:?}");
    }

    #[test]
    fn a_group_that_does_not_add_up_is_caught() {
        let tables = toml::parse(
            "[[group]]\nname = \"string\"\nexpected = 28\nstatus = \"shipped\"\n\n\
             [[command]]\nname = \"SET\"\ngroup = \"string\"\nsince = \"1.0.0\"\narity = -3\n\
             plan = \"point\"\nbounded = \"inherent\"\nstatus = \"shipped\"\nwire = \"none\"\n",
        )
        .unwrap();
        let mut bad = Vec::new();
        check_commands(&tables, &BTreeSet::new(), &mut bad);
        assert!(
            bad.iter().any(|b| b.contains("28 commands and lists 1")),
            "{bad:?}"
        );
    }

    /// A row for a command nobody has written does not count towards the group,
    /// which is what lets a command Redis added sit in the file honestly.
    #[test]
    fn a_planned_command_does_not_fill_a_groups_budget() {
        let head = "[[group]]\nname = \"set\"\nexpected = 2\nstatus = \"partial\"\n\n\
             [[command]]\nname = \"SUNION\"\ngroup = \"set\"\nsince = \"1.0.0\"\narity = -2\n\
             plan = \"whole-value\"\nbounded = \"risk\"\nstatus = \"shipped\"\nwire = \"none\"\n\n\
             [[command]]\nname = \"SUNIONCARD\"\ngroup = \"set\"\nsince = \"8.10.0\"\narity = -3\n\
             plan = \"whole-value\"\nbounded = \"inherent\"\nstatus = \"planned\"\nwire = \"none\"\n";
        let tables = toml::parse(head).unwrap();
        let mut bad = Vec::new();
        check_commands(&tables, &BTreeSet::new(), &mut bad);
        assert!(bad.is_empty(), "{bad:?}");
        // And the same two rows under a group claiming to be done are caught,
        // because one of the two is not.
        let tables = toml::parse(&head.replace("\"partial\"", "\"shipped\"")).unwrap();
        let mut bad = Vec::new();
        check_commands(&tables, &BTreeSet::new(), &mut bad);
        assert!(
            bad.iter().any(|b| b.contains("2 commands and lists 1")),
            "{bad:?}"
        );
    }

    /// A group being worked through says so, and says it with a number that is
    /// actually between the two ends.
    #[test]
    fn a_partial_group_that_is_really_finished_or_really_empty_is_caught() {
        for (expected, listed, want) in [(1, 1, "lists 1 of 1"), (9, 0, "lists 0 of 9")] {
            let mut text = format!(
                "[[group]]\nname = \"server\"\nexpected = {expected}\nstatus = \"partial\"\n\n"
            );
            for _ in 0..listed {
                text.push_str(
                    "[[command]]\nname = \"INFO\"\ngroup = \"server\"\nsince = \"1.0.0\"\n\
                     arity = -1\nplan = \"none\"\nbounded = \"inherent\"\nstatus = \"shipped\"\n\
                     wire = \"unverified\"\n\n",
                );
            }
            let tables = toml::parse(&text).unwrap();
            let mut bad = Vec::new();
            check_commands(&tables, &BTreeSet::new(), &mut bad);
            assert!(bad.iter().any(|b| b.contains(want)), "{bad:?}");
        }
    }

    /// The escape hatch for a command that touches no key stays where it
    /// belongs, or it is a way to ship anything without a plan.
    #[test]
    fn a_command_with_no_plan_outside_the_two_groups_is_caught() {
        let tables = toml::parse(
            "[[group]]\nname = \"string\"\nexpected = 1\nstatus = \"shipped\"\n\n\
             [[command]]\nname = \"GET\"\ngroup = \"string\"\nsince = \"1.0.0\"\narity = 2\n\
             plan = \"none\"\nbounded = \"inherent\"\nstatus = \"shipped\"\nwire = \"verified\"\n",
        )
        .unwrap();
        let mut bad = Vec::new();
        check_commands(&tables, &BTreeSet::new(), &mut bad);
        assert!(
            bad.iter().any(|b| b.contains("claims no storage plan")),
            "{bad:?}"
        );
    }

    /// The dispatch table and the file, in both directions.
    #[test]
    fn the_table_and_the_file_agree_about_every_command() {
        // A row that claims a verified argument order for a command nothing
        // runs, which is the claim that reads as done and is not.
        let tables = toml::parse(
            "[[command]]\nname = \"XCFGSET\"\ngroup = \"stream\"\nsince = \"8.8.0\"\narity = -4\n\
             plan = \"point\"\nbounded = \"inherent\"\nstatus = \"shipped\"\nwire = \"verified\"\n",
        )
        .unwrap();
        let mut bad = Vec::new();
        check_table(&tables, &mut bad);
        assert!(
            bad.iter().any(|b| b.contains("nothing dispatches it")),
            "{bad:?}"
        );
        // And an arity that drifted, which is the one that produces a client
        // that routes a command to the wrong place rather than an error.
        let tables = toml::parse(
            "[[command]]\nname = \"GET\"\ngroup = \"string\"\nsince = \"1.0.0\"\narity = 3\n\
             plan = \"point\"\nbounded = \"inherent\"\nstatus = \"shipped\"\nwire = \"verified\"\n",
        )
        .unwrap();
        let mut bad = Vec::new();
        check_table(&tables, &mut bad);
        assert!(
            bad.iter()
                .any(|b| b.contains("GET has arity 3 and the dispatch table has 2")),
            "{bad:?}"
        );
    }

    #[test]
    fn a_temporary_divergence_has_to_say_when_it_goes_away() {
        let tables = toml::parse(
            "[[divergence]]\nid = \"D-1\"\ntitle = \"t\"\nreason = \"r\"\nspec = \"s\"\n\
             status = \"until\"\n",
        )
        .unwrap();
        let mut bad = Vec::new();
        check_divergences(&tables, &mut bad);
        assert!(
            bad.iter()
                .any(|b| b.contains("which milestone resolves it")),
            "{bad:?}"
        );
    }
}
