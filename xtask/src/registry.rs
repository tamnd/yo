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
const PLANS: &[&str] = &["point", "cursor", "merge", "metadata", "whole-value"];
/// The bound or materialise verdicts.
const BOUNDED: &[&str] = &["inherent", "yes", "risk"];
/// Whether a command is implemented, and how far up it reaches.
const STATUSES: &[&str] = &["shipped", "planned"];
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
    bad
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
    let mut counted: Vec<(String, String)> = Vec::new();

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
        counted.push((group, name.clone()));

        one_of(t, "plan", PLANS, &name, bad);
        one_of(t, "bounded", BOUNDED, &name, bad);
        one_of(t, "wire", WIRE, &name, bad);
        let status = one_of(t, "status", STATUSES, &name, bad);

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
        if let Some(v) = t.get("divergent") {
            match v.as_str() {
                Some(id) if ids.contains(id) => {}
                Some(id) => bad.push(format!(
                    "commands.toml line {}: {name} names divergence {id}, which is not in divergences.toml",
                    t.line
                )),
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
fn check_group_budgets(groups: &[&Table], counted: &[(String, String)], bad: &mut Vec<String>) {
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
        if !STATUSES.contains(&status) {
            bad.push(format!(
                "commands.toml line {}: group {name} has status {status}, which is not one of {STATUSES:?}",
                g.line
            ));
        }
        let have = counted.iter().filter(|(gr, _)| *gr == name).count() as i64;
        if status == "shipped" && have != expected {
            bad.push(format!(
                "commands.toml line {}: group {name} says {expected} commands and lists {have}",
                g.line
            ));
        }
        if status == "planned" && have != 0 {
            bad.push(format!(
                "commands.toml line {}: group {name} is planned and already lists {have} commands, so it is shipped",
                g.line
            ));
        }
    }
    for (group, name) in counted {
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
