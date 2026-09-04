//! Every index on the server, and the names that point at them.

use crate::index::Index;

/// What went wrong with a name, in the terms the caller has to answer in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Clash {
    /// There is already an index or an alias with this name.
    Taken,
    /// There is no index with this name.
    Missing,
    /// The name is an alias that already points somewhere.
    Aliased,
    /// The name is an index, so it cannot also be an alias.
    IsIndex,
    /// The index being pointed at is itself an alias, which is a name an alias
    /// may not be hung off even though every other command takes it happily.
    IsAlias,
}

/// An alias and the index name it points at, in that order.
type Pointer = (Box<[u8]>, Box<[u8]>);

/// Every index the server holds and every alias pointing at one.
///
/// One of these per server and not one per database, which is worth saying out
/// loud because every other collection in this build is per database. A real
/// server keeps the search indexes in the module, the module has one table, and
/// `SELECT 1` followed by `FT._LIST` answers with the indexes made on database
/// zero. Putting it per database here would answer an empty list there, which
/// is the kind of difference that only shows up in somebody's failover.
#[derive(Debug, Default)]
pub struct Registry {
    /// The indexes, in the order they were created.
    ///
    /// A vector and not a map. A server has tens of indexes rather than
    /// millions, a lookup is a walk over a handful of short names, and the
    /// order is what `FT._LIST` answers with. A map would answer in whatever
    /// order its buckets happened to be in, which is what a real server does
    /// and is a worse answer than a stable one.
    indexes: Vec<Index>,
    /// Alias to index name, in the order the aliases were added.
    aliases: Vec<Pointer>,
}

impl Registry {
    /// A server with no indexes on it.
    #[must_use]
    pub fn new() -> Registry {
        Registry::default()
    }

    /// How many indexes there are.
    #[must_use]
    pub fn len(&self) -> usize {
        self.indexes.len()
    }

    /// Whether there are none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indexes.is_empty()
    }

    /// Every index, in the order they were created.
    pub fn iter(&self) -> impl Iterator<Item = &Index> {
        self.indexes.iter()
    }

    /// The index with this exact name, ignoring aliases.
    #[must_use]
    pub fn named(&self, name: &[u8]) -> Option<&Index> {
        self.indexes.iter().find(|i| &*i.name == name)
    }

    /// The index a client means by a name, which is an index name first and an
    /// alias second.
    ///
    /// That order is not arbitrary. `FT.CREATE` over an existing alias makes a
    /// real index with that name, and from then on the name means the index,
    /// which is what a real server does with it.
    #[must_use]
    pub fn get(&self, name: &[u8]) -> Option<&Index> {
        self.named(name).or_else(|| {
            let real = self.target(name)?;
            self.named(real)
        })
    }

    /// The same, with the index borrowed mutably.
    pub fn get_mut(&mut self, name: &[u8]) -> Option<&mut Index> {
        let real: Box<[u8]> = match self.target(name) {
            Some(r) if self.named(name).is_none() => r.into(),
            _ => name.into(),
        };
        self.indexes.iter_mut().find(|i| i.name == real)
    }

    /// The index a client means by a name, counted as one use of it.
    ///
    /// The counter `FT.INFO` reports is a count of how many times the spec has
    /// been looked up by name rather than a count of anything a query did, so
    /// every command that resolves a name goes through here and every command
    /// that fails before it gets that far does not. A command that resolves the
    /// name and then refuses the arguments after it has still counted, which is
    /// what a real server does and is the only way to see the difference.
    pub fn open(&mut self, name: &[u8]) -> Option<&mut Index> {
        let index = self.get_mut(name)?;
        index.uses += 1;
        Some(index)
    }

    /// The same for a command that only needs to know the index is there.
    pub fn touch(&mut self, name: &[u8]) -> bool {
        self.open(name).is_some()
    }

    /// Where an alias points, or `None` if it is not an alias.
    #[must_use]
    pub fn target(&self, alias: &[u8]) -> Option<&[u8]> {
        self.aliases
            .iter()
            .find(|(a, _)| &**a == alias)
            .map(|(_, i)| &**i)
    }

    /// Every alias pointing at an index, in the order they were added.
    pub fn aliases_of<'a>(&'a self, name: &'a [u8]) -> impl Iterator<Item = &'a [u8]> {
        self.aliases
            .iter()
            .filter(move |(_, i)| &**i == name)
            .map(|(a, _)| &**a)
    }

    /// Puts an index in, or says the name is taken.
    ///
    /// A name that is already an alias is taken too, since `FT.SEARCH` would
    /// have two things to mean by it.
    ///
    /// # Errors
    ///
    /// [`Clash::Taken`] when an index of that name is already here.
    pub fn create(&mut self, index: Index) -> Result<(), Clash> {
        if self.named(&index.name).is_some() {
            return Err(Clash::Taken);
        }
        self.indexes.push(index);
        Ok(())
    }

    /// Takes an index out, along with every alias that pointed at it.
    ///
    /// The aliases go with it because an alias to an index that is gone is a
    /// name that answers "index not found" from a table that says it exists,
    /// and the next `FT.CREATE` of the same name would silently inherit them.
    ///
    /// # Errors
    ///
    /// [`Clash::Missing`] when there is no such index.
    pub fn drop(&mut self, name: &[u8]) -> Result<Index, Clash> {
        let real: Box<[u8]> = self.get(name).ok_or(Clash::Missing)?.name.clone();
        let at = self
            .indexes
            .iter()
            .position(|i| i.name == real)
            .ok_or(Clash::Missing)?;
        self.aliases.retain(|(_, i)| *i != real);
        Ok(self.indexes.remove(at))
    }

    /// Points a new alias at an index.
    ///
    /// # Errors
    ///
    /// [`Clash::IsIndex`] when the alias is the name of an index, which would
    /// leave the name meaning two things. [`Clash::Aliased`] when the alias is
    /// already pointing somewhere. [`Clash::IsAlias`] when the index being
    /// pointed at is an alias. [`Clash::Missing`] when the index is not here.
    pub fn alias(&mut self, alias: &[u8], name: &[u8]) -> Result<(), Clash> {
        if self.named(alias).is_some() {
            return Err(Clash::IsIndex);
        }
        if self.target(alias).is_some() {
            return Err(Clash::Aliased);
        }
        let real = self.point_at(name)?;
        self.aliases.push((alias.into(), real));
        Ok(())
    }

    /// The index name an alias may be hung off, which is an index name and never
    /// another alias.
    ///
    /// Every other command that takes an index name takes an alias in its place
    /// and follows it. These two do not, so a chain of aliases cannot be built
    /// and there is nothing to follow at query time beyond one hop.
    fn point_at(&self, name: &[u8]) -> Result<Box<[u8]>, Clash> {
        match self.named(name) {
            Some(index) => Ok(index.name.clone()),
            None if self.target(name).is_some() => Err(Clash::IsAlias),
            None => Err(Clash::Missing),
        }
    }

    /// Moves an alias to another index, adding it if it was not there.
    ///
    /// # Errors
    ///
    /// The same as [`Registry::alias`] minus [`Clash::Aliased`], since moving
    /// an alias that is already pointing somewhere is the whole point.
    pub fn realias(&mut self, alias: &[u8], name: &[u8]) -> Result<(), Clash> {
        if self.named(alias).is_some() {
            return Err(Clash::IsIndex);
        }
        let real = self.point_at(name)?;
        self.aliases.retain(|(a, _)| **a != *alias);
        self.aliases.push((alias.into(), real));
        Ok(())
    }

    /// Takes an alias out.
    ///
    /// # Errors
    ///
    /// [`Clash::Missing`] when there is no such alias.
    pub fn unalias(&mut self, alias: &[u8]) -> Result<(), Clash> {
        let at = self
            .aliases
            .iter()
            .position(|(a, _)| &**a == alias)
            .ok_or(Clash::Missing)?;
        self.aliases.remove(at);
        Ok(())
    }

    /// Throws every index and alias away, which is what emptying the keyspace
    /// does to them.
    pub fn clear(&mut self) {
        self.indexes.clear();
        self.aliases.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{Field, Kind, Text};
    use crate::index::{Definition, Index};

    fn index(name: &str) -> Index {
        Index::new(
            name.as_bytes(),
            Definition::default(),
            vec![Field::new(b"t", Kind::Text(Text::default()))],
        )
    }

    #[test]
    fn a_name_can_only_be_used_once() {
        let mut r = Registry::new();
        assert_eq!(r.create(index("a")), Ok(()));
        assert_eq!(r.create(index("a")), Err(Clash::Taken));
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn an_alias_reaches_the_index_it_points_at() {
        let mut r = Registry::new();
        r.create(index("a")).unwrap();
        r.alias(b"al", b"a").unwrap();
        assert_eq!(r.get(b"al").map(|i| &*i.name), Some(&b"a"[..]));
        assert_eq!(r.get_mut(b"al").map(|i| &*i.name), Some(&b"a"[..]));
        assert_eq!(r.aliases_of(b"a").collect::<Vec<_>>(), vec![&b"al"[..]]);
    }

    /// An index name beats an alias of the same name, because `FT.CREATE` over
    /// an alias is allowed and the name has to mean one thing afterwards.
    #[test]
    fn a_real_name_beats_an_alias() {
        let mut r = Registry::new();
        r.create(index("a")).unwrap();
        r.alias(b"al", b"a").unwrap();
        r.create(index("al")).unwrap();
        assert_eq!(r.get(b"al").map(|i| &*i.name), Some(&b"al"[..]));
    }

    #[test]
    fn an_alias_cannot_take_an_index_name_or_be_taken_twice() {
        let mut r = Registry::new();
        r.create(index("a")).unwrap();
        r.create(index("b")).unwrap();
        assert_eq!(r.alias(b"a", b"b"), Err(Clash::IsIndex));
        r.alias(b"al", b"a").unwrap();
        assert_eq!(r.alias(b"al", b"b"), Err(Clash::Aliased));
        assert_eq!(r.alias(b"al2", b"nope"), Err(Clash::Missing));
    }

    #[test]
    fn moving_an_alias_leaves_one_of_it() {
        let mut r = Registry::new();
        r.create(index("a")).unwrap();
        r.create(index("b")).unwrap();
        r.alias(b"al", b"a").unwrap();
        r.realias(b"al", b"b").unwrap();
        assert_eq!(r.target(b"al"), Some(&b"b"[..]));
        assert_eq!(r.aliases_of(b"b").count(), 1);
        assert_eq!(r.aliases_of(b"a").count(), 0);
    }

    /// The aliases go with the index, so the name that pointed at it stops
    /// answering rather than pointing at nothing.
    #[test]
    fn dropping_an_index_takes_its_aliases_with_it() {
        let mut r = Registry::new();
        r.create(index("a")).unwrap();
        r.alias(b"al", b"a").unwrap();
        assert_eq!(r.drop(b"al").map(|i| i.name), Ok(b"a".to_vec().into()));
        assert!(r.is_empty());
        assert_eq!(r.target(b"al"), None);
        assert_eq!(r.drop(b"a").err(), Some(Clash::Missing));
    }

    /// Every other command follows an alias, and these two do not, so there is
    /// never a chain of them to walk.
    #[test]
    fn an_alias_cannot_point_at_another_alias() {
        let mut r = Registry::new();
        r.create(index("a")).unwrap();
        r.alias(b"al", b"a").unwrap();
        assert_eq!(r.alias(b"al2", b"al"), Err(Clash::IsAlias));
        assert_eq!(r.realias(b"al2", b"al"), Err(Clash::IsAlias));
        assert_eq!(r.realias(b"al2", b"nope"), Err(Clash::Missing));
        assert_eq!(r.target(b"al2"), None);
    }

    /// The counter is a count of lookups by name, so it moves for the commands
    /// that resolve a name and stays put for the ones that never get that far.
    #[test]
    fn opening_an_index_counts_a_use_of_it() {
        let mut r = Registry::new();
        r.create(index("a")).unwrap();
        r.alias(b"al", b"a").unwrap();
        assert_eq!(r.named(b"a").map(|i| i.uses), Some(0));
        assert!(r.touch(b"a"));
        // Through an alias it is still one lookup and not two.
        assert!(r.touch(b"al"));
        assert_eq!(r.named(b"a").map(|i| i.uses), Some(2));
        assert!(!r.touch(b"nope"));
        assert_eq!(r.open(b"a").map(|i| i.uses), Some(3));
    }

    #[test]
    fn an_alias_can_be_taken_out_on_its_own() {
        let mut r = Registry::new();
        r.create(index("a")).unwrap();
        r.alias(b"al", b"a").unwrap();
        r.unalias(b"al").unwrap();
        assert_eq!(r.unalias(b"al"), Err(Clash::Missing));
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn emptying_the_keyspace_empties_this_too() {
        let mut r = Registry::new();
        r.create(index("a")).unwrap();
        r.alias(b"al", b"a").unwrap();
        r.clear();
        assert!(r.is_empty());
        assert_eq!(r.target(b"al"), None);
    }
}
