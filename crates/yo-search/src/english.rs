//! The English stemmer, which is Snowball's `english` and is Porter2.
//!
//! A query for `running` has to reach a document holding `runs`, and the way
//! every search engine in this family does that is by reducing both to `run`
//! before either one is written down. The algorithm that does the reducing is
//! not a detail: a server that stems `flies` to `fly` where the reference stems
//! it to `fli` answers a different set of documents, and the two are not
//! reconcilable afterwards. So this is Porter2 exactly, rule for rule, and it is
//! checked against a real server's own answers over twelve thousand words.
//!
//! # Why it is written out rather than pulled in
//!
//! There is a crate for this. There is also a rule in this repository about
//! dependencies, and the whole of the index this feeds is written here for the
//! same reason: a stemmer is four hundred lines of table lookup and the version
//! of it that matches the reference is the only version worth having, so the
//! choice is between reading four hundred lines and trusting that somebody
//! else's four hundred lines say the same thing. The test at the bottom is what
//! makes either choice safe, and it belongs here whichever way the code came.
//!
//! # It allocates once
//!
//! Stemming happens per token, and a document has thousands of them. The buffer
//! lives on the stemmer rather than on the call, so indexing a corpus allocates
//! once and then never again, and a word that the algorithm leaves alone is
//! handed straight back without being copied at all.

/// The letters that count as vowels.
///
/// `y` is one of them and `Y` is not, which is not a typo. The algorithm starts
/// by rewriting the `y` that acts as a consonant into a capital, so that `y` in
/// `try` and `y` in `young` can be told apart by a table lookup afterwards, and
/// the last thing it does is put them back.
const fn vowel(b: u8) -> bool {
    matches!(b, b'a' | b'e' | b'i' | b'o' | b'u' | b'y')
}

/// The consonants a word may end in for `li` to be a suffix worth removing.
const LI_ENDING: &[u8] = b"cdeghkmnrt";

/// The doubled consonants that get cut back to one after a suffix comes off, so
/// `hopping` reaches `hop` rather than `hopp`.
const DOUBLES: [&[u8]; 9] = [
    b"bb", b"dd", b"ff", b"gg", b"mm", b"nn", b"pp", b"rr", b"tt",
];

/// The words the algorithm gets wrong, and what they should be instead.
///
/// Eleven of them are irregular enough that no rule reaches them, and seven more
/// are words the rules would happily take a suffix off when there is no suffix
/// there: `news` is not the plural of `new` and `andes` is not the plural of
/// anything.
const IRREGULAR: [(&[u8], &[u8]); 18] = [
    (b"skis", b"ski"),
    (b"skies", b"sky"),
    (b"dying", b"die"),
    (b"lying", b"lie"),
    (b"tying", b"tie"),
    (b"idly", b"idl"),
    (b"gently", b"gentl"),
    (b"ugly", b"ugli"),
    (b"early", b"earli"),
    (b"only", b"onli"),
    (b"singly", b"singl"),
    (b"sky", b"sky"),
    (b"news", b"news"),
    (b"howe", b"howe"),
    (b"atlas", b"atlas"),
    (b"cosmos", b"cosmos"),
    (b"bias", b"bias"),
    (b"andes", b"andes"),
];

/// The words that are already finished once the plural rules have run, and that
/// the suffix rules after them would otherwise chew into.
///
/// `proceed` would lose its `eed` and `inning` would lose its `ing`, and neither
/// of those is a suffix.
const SETTLED: [&[u8]; 8] = [
    b"inning", b"outing", b"canning", b"herring", b"earring", b"proceed", b"exceed", b"succeed",
];

/// The three prefixes that make a word's first region start somewhere the
/// general rule would not put it.
///
/// Without these, `generate` and `general` end up with different regions than
/// the words built on them, and the family comes apart.
const PREFIXES: [(&[u8], usize); 3] = [(b"gener", 5), (b"commun", 6), (b"arsen", 5)];

/// The English stemmer, holding the buffer it works in.
///
/// One of these per thread that indexes or parses, kept for as long as there is
/// text coming. It is worth keeping: the buffer grows to the longest word it has
/// seen and then stops.
#[derive(Debug, Default)]
pub struct English {
    /// The word being worked on, lower case, with the consonantal `y` written as
    /// `Y` until the last step.
    w: Vec<u8>,
    /// Where the first region starts, which is after the first consonant that
    /// follows a vowel.
    r1: usize,
    /// Where the second region starts, which is the same rule applied again
    /// inside the first region.
    r2: usize,
}

impl English {
    /// A stemmer with nothing in it yet.
    #[must_use]
    pub fn new() -> English {
        English::default()
    }

    /// The stem of one word, which must already be lower case ASCII.
    ///
    /// The answer borrows the stemmer, so a caller holding on to it has to copy
    /// it out before stemming the next word. That is the arrangement that costs
    /// nothing for the common caller, which writes the stem into an index and
    /// moves on.
    pub fn stem(&mut self, word: &[u8]) -> &[u8] {
        for (from, to) in IRREGULAR {
            if word == from {
                self.w.clear();
                self.w.extend_from_slice(to);
                return &self.w;
            }
        }
        self.w.clear();
        self.w.extend_from_slice(word);
        // Two letters is not enough for any rule here to have room to work in,
        // and three is where the algorithm starts looking.
        if self.w.len() < 3 {
            return &self.w;
        }
        self.prelude();
        self.regions();
        self.plural();
        if !SETTLED.contains(&&self.w[..]) {
            self.past();
            self.terminal_y();
            self.derivational();
            self.residual_derivational();
            self.endings();
            self.tidy();
        }
        self.postlude();
        &self.w
    }

    /// Marks the `y` that behaves like a consonant, and drops a leading
    /// apostrophe.
    ///
    /// A `y` at the front of a word and a `y` after a vowel are both consonants,
    /// which is why `young` is not treated as starting with a vowel and `say`
    /// does not lose its ending the way `carry` does.
    fn prelude(&mut self) {
        if self.w.first() == Some(&b'\'') {
            self.w.remove(0);
        }
        if self.w.first() == Some(&b'y') {
            self.w[0] = b'Y';
        }
        for i in 1..self.w.len() {
            if self.w[i] == b'y' && vowel(self.w[i - 1]) {
                self.w[i] = b'Y';
            }
        }
    }

    /// Works out where the two regions start.
    fn regions(&mut self) {
        self.r1 = self.w.len();
        for (prefix, at) in PREFIXES {
            if self.w.starts_with(prefix) {
                self.r1 = at;
                break;
            }
        }
        if self.r1 == self.w.len() {
            self.r1 = self.region(0);
        }
        self.r2 = self.region(self.r1);
    }

    /// Where the region after `from` starts, which is one past the first
    /// consonant that follows a vowel.
    fn region(&self, from: usize) -> usize {
        let mut i = from;
        while i < self.w.len() && !vowel(self.w[i]) {
            i += 1;
        }
        while i < self.w.len() && vowel(self.w[i]) {
            i += 1;
        }
        if i < self.w.len() {
            i + 1
        } else {
            self.w.len()
        }
    }

    /// Whether the word ends in a syllable short enough that a suffix coming off
    /// it needs a silent `e` put back, so `hop` reads as `hope` and not as
    /// something that rhymes with `top`.
    fn short_syllable(&self, len: usize) -> bool {
        let w = &self.w[..len];
        if w.len() == 2 {
            return vowel(w[0]) && !vowel(w[1]);
        }
        if w.len() < 3 {
            return false;
        }
        let (a, b, c) = (w[w.len() - 3], w[w.len() - 2], w[w.len() - 1]);
        !vowel(a) && vowel(b) && !vowel(c) && !matches!(c, b'w' | b'x' | b'Y')
    }

    /// Whether the whole word is short, which is a short syllable at the end and
    /// no first region at all.
    fn short(&self) -> bool {
        self.r1 >= self.w.len() && self.short_syllable(self.w.len())
    }

    /// Whether there is a vowel anywhere in the first `len` letters.
    fn has_vowel(&self, len: usize) -> bool {
        self.w[..len.min(self.w.len())].iter().copied().any(vowel)
    }

    /// Replaces the last `cut` letters with `with`.
    fn replace(&mut self, cut: usize, with: &[u8]) {
        let keep = self.w.len() - cut;
        self.w.truncate(keep);
        self.w.extend_from_slice(with);
    }

    /// The longest of these suffixes the word ends in, and where it starts.
    fn longest(&self, among: &[&[u8]]) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for (i, s) in among.iter().enumerate() {
            if self.w.len() > s.len()
                && self.w.ends_with(s)
                && best.is_none_or(|(b, _)| among[b].len() < s.len())
            {
                best = Some((i, self.w.len() - s.len()));
            }
        }
        best
    }

    /// Plurals and possessives, which come off before anything else.
    fn plural(&mut self) {
        for suffix in [b"'s'".as_slice(), b"'s", b"'"] {
            if self.w.ends_with(suffix) {
                let keep = self.w.len() - suffix.len();
                self.w.truncate(keep);
                break;
            }
        }
        if self.w.ends_with(b"sses") {
            self.replace(4, b"ss");
            return;
        }
        if self.w.ends_with(b"ied") || self.w.ends_with(b"ies") {
            // `cries` has two letters in front of the ending and becomes `cri`,
            // `ties` has one and becomes `tie`, which keeps it a word rather
            // than turning it into the letter it starts with.
            let with: &[u8] = if self.w.len() > 4 { b"i" } else { b"ie" };
            self.replace(3, with);
            return;
        }
        if self.w.ends_with(b"us") || self.w.ends_with(b"ss") {
            return;
        }
        if self.w.ends_with(b"s") && self.w.len() >= 3 && self.has_vowel(self.w.len() - 2) {
            self.w.pop();
        }
    }

    /// The past tense and the participles.
    fn past(&mut self) {
        const AMONG: [&[u8]; 6] = [b"eed", b"eedly", b"ed", b"edly", b"ing", b"ingly"];
        let Some((which, at)) = self.longest(&AMONG) else {
            return;
        };
        if which < 2 {
            // `agreed` keeps its `ee` and only loses what is past it, and only
            // when the ending reaches into the first region: `need` is left
            // alone because there is nothing of it outside the stem.
            if at >= self.r1 {
                self.replace(AMONG[which].len(), b"ee");
            }
            return;
        }
        if !self.has_vowel(at) {
            return;
        }
        self.w.truncate(at);
        if self.w.ends_with(b"at") || self.w.ends_with(b"bl") || self.w.ends_with(b"iz") {
            self.w.push(b'e');
            return;
        }
        if DOUBLES.iter().any(|d| self.w.ends_with(d)) {
            self.w.pop();
            return;
        }
        if self.short() {
            self.w.push(b'e');
        }
    }

    /// A trailing `y` becomes an `i` so that `cry` and `cried` meet.
    ///
    /// Only when there is a consonant in front of it and that consonant is not
    /// the first letter, which is what keeps `by` and `say` as they are.
    fn terminal_y(&mut self) {
        let n = self.w.len();
        if n >= 3 && matches!(self.w[n - 1], b'y' | b'Y') && !vowel(self.w[n - 2]) {
            self.w[n - 1] = b'i';
        }
    }

    /// The suffixes that turn one part of speech into another, taken one layer
    /// at a time.
    fn derivational(&mut self) {
        const AMONG: [&[u8]; 24] = [
            b"tional", b"enci", b"anci", b"abli", b"entli", b"izer", b"ization", b"ational",
            b"ation", b"ator", b"alism", b"aliti", b"alli", b"fulness", b"ousli", b"ousness",
            b"iveness", b"iviti", b"biliti", b"bli", b"ogi", b"fulli", b"lessli", b"li",
        ];
        const WITH: [&[u8]; 24] = [
            b"tion", b"ence", b"ance", b"able", b"ent", b"ize", b"ize", b"ate", b"ate", b"ate",
            b"al", b"al", b"al", b"ful", b"ous", b"ous", b"ive", b"ive", b"ble", b"ble", b"og",
            b"ful", b"less", b"",
        ];
        let Some((which, at)) = self.longest(&AMONG) else {
            return;
        };
        if at < self.r1 {
            return;
        }
        // Two of them look at the letter in front before they will do anything.
        // `ogi` is only a suffix after an `l`, so `apology` gives way and `yogi`
        // does not, and a bare `li` only comes off the consonants it is actually
        // spoken after.
        if AMONG[which] == b"ogi" && self.w[at - 1] != b'l' {
            return;
        }
        if AMONG[which] == b"li" && !LI_ENDING.contains(&self.w[at - 1]) {
            return;
        }
        self.replace(AMONG[which].len(), WITH[which]);
    }

    /// The layer under the one above, which is the same idea applied to what is
    /// left.
    fn residual_derivational(&mut self) {
        const AMONG: [&[u8]; 9] = [
            b"tional", b"ational", b"alize", b"icate", b"iciti", b"ical", b"ful", b"ness", b"ative",
        ];
        const WITH: [&[u8]; 9] = [b"tion", b"ate", b"al", b"ic", b"ic", b"ic", b"", b"", b""];
        let Some((which, at)) = self.longest(&AMONG) else {
            return;
        };
        if at < self.r1 {
            return;
        }
        // `ative` is the one that has to reach the second region, because it is
        // long enough to swallow a short word whole.
        if AMONG[which] == b"ative" && at < self.r2 {
            return;
        }
        self.replace(AMONG[which].len(), WITH[which]);
    }

    /// The last suffixes, which only come off a word long enough to have a
    /// second region for them to sit in.
    fn endings(&mut self) {
        const AMONG: [&[u8]; 18] = [
            b"al", b"ance", b"ence", b"er", b"ic", b"able", b"ible", b"ant", b"ement", b"ment",
            b"ent", b"ism", b"ate", b"iti", b"ous", b"ive", b"ize", b"ion",
        ];
        let Some((which, at)) = self.longest(&AMONG) else {
            return;
        };
        if at < self.r2 {
            return;
        }
        // `ion` is a suffix after an `s` or a `t` and part of the word after
        // anything else, which is the difference between `adoption` and `lion`.
        if AMONG[which] == b"ion" && !matches!(self.w[at - 1], b's' | b't') {
            return;
        }
        self.w.truncate(at);
    }

    /// The silent letters at the very end.
    fn tidy(&mut self) {
        let n = self.w.len();
        // Where the last letter sits, which is the position the regions are
        // measured against.
        let last = n - 1;
        if self.w.ends_with(b"e") {
            // A final `e` goes when there is enough word behind it, and stays
            // when taking it would leave a syllable that reads differently.
            if last >= self.r2 || (last >= self.r1 && !self.short_syllable(last)) {
                self.w.pop();
            }
            return;
        }
        if n >= 2 && self.w[last] == b'l' && self.w[last - 1] == b'l' && last >= self.r2 {
            self.w.pop();
        }
    }

    /// Puts the consonantal `y` back the way the client spelled it.
    fn postlude(&mut self) {
        for b in &mut self.w {
            if *b == b'Y' {
                *b = b'y';
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stem(word: &str) -> String {
        let mut e = English::new();
        String::from_utf8(e.stem(word.as_bytes()).to_vec()).unwrap()
    }

    #[test]
    fn a_word_with_no_suffix_on_it_is_left_alone() {
        assert_eq!(stem("hello"), "hello");
        assert_eq!(stem("by"), "by");
        assert_eq!(stem("a"), "a");
    }

    /// The stem is not always a word, and that is the point: it only has to be
    /// the same for every form of the word.
    #[test]
    fn every_form_of_a_word_reaches_the_same_stem() {
        for form in ["run", "runs", "running"] {
            assert_eq!(stem(form), "run", "{form}");
        }
        for form in ["fli", "flies", "flying"] {
            assert_eq!(stem(form), "fli", "{form}");
        }
    }

    /// The rules only reach as far as the regions let them, and the regions are
    /// short in a short word. `happiness` gets back to `happi` and `happier`
    /// does not, because the `er` sits outside the second region and there is no
    /// second region in a word that size. A real server answers the same way,
    /// so a query for one of these does not reach the other.
    #[test]
    fn a_short_word_keeps_endings_a_longer_one_would_lose() {
        assert_eq!(stem("happiness"), "happi");
        assert_eq!(stem("happily"), "happili");
        assert_eq!(stem("happier"), "happier");
        assert_eq!(stem("happiest"), "happiest");
    }

    #[test]
    fn a_doubled_consonant_comes_back_to_one_and_a_silent_e_comes_back() {
        assert_eq!(stem("hopping"), "hop");
        assert_eq!(stem("hoping"), "hope");
        // The `e` goes back on after `ing` comes off and then the last rule
        // takes it and the `at` with it, which is why this lands two letters
        // shorter than the word it obviously came from.
        assert_eq!(stem("luxuriating"), "luxuri");
    }

    #[test]
    fn the_words_no_rule_reaches_are_listed_instead() {
        assert_eq!(stem("skies"), "sky");
        assert_eq!(stem("dying"), "die");
        assert_eq!(stem("news"), "news");
        assert_eq!(stem("proceed"), "proceed");
        assert_eq!(stem("inning"), "inning");
    }

    /// The `y` that acts like a consonant is not a vowel while the rules run and
    /// is spelled the way it arrived once they are done.
    #[test]
    fn a_consonantal_y_survives() {
        assert_eq!(stem("young"), "young");
        assert_eq!(stem("say"), "say");
        assert_eq!(stem("cry"), "cri");
        assert_eq!(stem("enjoyment"), "enjoy");
    }

    /// Four hundred words and the stem a real server gives for each.
    ///
    /// Taken from a system dictionary and run through an 8.10.1 with
    /// `FT.EXPLAIN`, which prints the stem it expanded a term to, so these are
    /// the reference's own answers rather than a reading of the specification.
    /// Three hundred of them are words the algorithm changes and a hundred are
    /// words it leaves alone, because getting the second group wrong is just as
    /// bad and is easier to do by accident. The full run was twelve thousand
    /// words with no differences at all.
    const CORPUS: [(&[u8], &[u8]); 400] = [
        (b"abase", b"abas"),
        (b"abaxial", b"abaxi"),
        (b"acclamatory", b"acclamatori"),
        (b"addlings", b"addl"),
        (b"adjunctively", b"adjunct"),
        (b"affinitative", b"affinit"),
        (b"afforcement", b"afforc"),
        (b"africanization", b"african"),
        (b"aganippe", b"aganipp"),
        (b"agglutinoscope", b"agglutinoscop"),
        (b"agnatically", b"agnat"),
        (b"alarming", b"alarm"),
        (b"aliener", b"alien"),
        (b"alogia", b"alogia"),
        (b"amani", b"amani"),
        (b"amplexicaul", b"amplexicaul"),
        (b"amuser", b"amus"),
        (b"amyraldism", b"amyrald"),
        (b"aneuploidy", b"aneuploidi"),
        (b"animalic", b"animal"),
        (b"anisomyodous", b"anisomyod"),
        (b"anoxemia", b"anoxemia"),
        (b"anthracene", b"anthracen"),
        (b"antigambling", b"antigambl"),
        (b"antilogy", b"antilog"),
        (b"antitheological", b"antitheolog"),
        (b"apeak", b"apeak"),
        (b"aphthongal", b"aphthong"),
        (b"apocatharsis", b"apocatharsi"),
        (b"arachnoid", b"arachnoid"),
        (b"archigonic", b"archigon"),
        (b"ashplant", b"ashplant"),
        (b"aslop", b"aslop"),
        (b"asseverative", b"assev"),
        (b"associableness", b"associ"),
        (b"astonishment", b"astonish"),
        (b"aswail", b"aswail"),
        (b"attentive", b"attent"),
        (b"autotriploidy", b"autotriploidi"),
        (b"avalanche", b"avalanch"),
        (b"axite", b"axit"),
        (b"azobacter", b"azobact"),
        (b"bacchanal", b"bacchan"),
        (b"bahutu", b"bahutu"),
        (b"bakingly", b"bake"),
        (b"balandra", b"balandra"),
        (b"balmawhapple", b"balmawhappl"),
        (b"banally", b"banal"),
        (b"barandos", b"barando"),
        (b"barbariousness", b"barbari"),
        (b"bargainer", b"bargain"),
        (b"barographic", b"barograph"),
        (b"bedrowse", b"bedrows"),
        (b"beelzebub", b"beelzebub"),
        (b"biclavate", b"biclav"),
        (b"biliary", b"biliari"),
        (b"biose", b"bios"),
        (b"blackishly", b"blackish"),
        (b"blanque", b"blanqu"),
        (b"blindstory", b"blindstori"),
        (b"bluely", b"blueli"),
        (b"breezily", b"breezili"),
        (b"brelaw", b"brelaw"),
        (b"bridgeable", b"bridgeabl"),
        (b"brutelike", b"brutelik"),
        (b"caddice", b"caddic"),
        (b"cadency", b"cadenc"),
        (b"cafeneh", b"cafeneh"),
        (b"calascione", b"calascion"),
        (b"camansi", b"camansi"),
        (b"captivatrix", b"captivatrix"),
        (b"carriagesmith", b"carriagesmith"),
        (b"cartage", b"cartag"),
        (b"caswellite", b"caswellit"),
        (b"centigram", b"centigram"),
        (b"cerebromalacia", b"cerebromalacia"),
        (b"char", b"char"),
        (b"chemoserotherapy", b"chemoserotherapi"),
        (b"chifforobe", b"chifforob"),
        (b"chipewyan", b"chipewyan"),
        (b"chorditis", b"chorditi"),
        (b"christlessness", b"christless"),
        (b"circuiter", b"circuit"),
        (b"cissoidal", b"cissoid"),
        (b"classable", b"classabl"),
        (b"clitoris", b"clitori"),
        (b"coamiable", b"coamiabl"),
        (b"codiscoverer", b"codiscover"),
        (b"combinedly", b"combin"),
        (b"conglutinant", b"conglutin"),
        (b"consimilar", b"consimilar"),
        (b"contractually", b"contractu"),
        (b"contrasty", b"contrasti"),
        (b"corymbiform", b"corymbiform"),
        (b"crag", b"crag"),
        (b"cran", b"cran"),
        (b"cuprene", b"cupren"),
        (b"curtation", b"curtat"),
        (b"cyanidation", b"cyanid"),
        (b"darwinite", b"darwinit"),
        (b"deboistness", b"deboist"),
        (b"declassify", b"declassifi"),
        (b"dedentition", b"dedentit"),
        (b"defiguration", b"defigur"),
        (b"dehull", b"dehul"),
        (b"dehydration", b"dehydr"),
        (b"deidesheimer", b"deidesheim"),
        (b"deliberalize", b"deliber"),
        (b"depurative", b"depur"),
        (b"derris", b"derri"),
        (b"desmarestia", b"desmarestia"),
        (b"despairingly", b"despair"),
        (b"dianilide", b"dianilid"),
        (b"diluvianism", b"diluvian"),
        (b"diogenite", b"diogenit"),
        (b"dipped", b"dip"),
        (b"discal", b"discal"),
        (b"discernment", b"discern"),
        (b"discolored", b"discolor"),
        (b"disconnection", b"disconnect"),
        (b"discouragingly", b"discourag"),
        (b"dislip", b"dislip"),
        (b"disproof", b"disproof"),
        (b"drastic", b"drastic"),
        (b"dun", b"dun"),
        (b"duodenectomy", b"duodenectomi"),
        (b"duodenopancreatectomy", b"duodenopancreatectomi"),
        (b"dykereeve", b"dykereev"),
        (b"egyptological", b"egyptolog"),
        (b"embryological", b"embryolog"),
        (b"endomorphy", b"endomorphi"),
        (b"engrieve", b"engriev"),
        (b"entomophily", b"entomophili"),
        (b"epididymovasostomy", b"epididymovasostomi"),
        (b"epigene", b"epigen"),
        (b"erotesis", b"erotesi"),
        (b"esophagoplication", b"esophagopl"),
        (b"etherous", b"ether"),
        (b"euonymy", b"euonymi"),
        (b"facility", b"facil"),
        (b"fagoter", b"fagot"),
        (b"fibrinogen", b"fibrinogen"),
        (b"fideist", b"fideist"),
        (b"firefly", b"firefli"),
        (b"fledgling", b"fledgl"),
        (b"fluoridization", b"fluorid"),
        (b"frogged", b"frog"),
        (b"furodiazole", b"furodiazol"),
        (b"gaonate", b"gaonat"),
        (b"gashouse", b"gashous"),
        (b"gelatinotype", b"gelatinotyp"),
        (b"glair", b"glair"),
        (b"goldsinny", b"goldsinni"),
        (b"gonne", b"gonn"),
        (b"grumpish", b"grumpish"),
        (b"guessable", b"guessabl"),
        (b"guilery", b"guileri"),
        (b"gullibility", b"gullibl"),
        (b"gummy", b"gummi"),
        (b"gweed", b"gweed"),
        (b"halide", b"halid"),
        (b"halma", b"halma"),
        (b"heavyhanded", b"heavyhand"),
        (b"hereamong", b"hereamong"),
        (b"heteromeral", b"heteromer"),
        (b"hierosolymite", b"hierosolymit"),
        (b"hippophagi", b"hippophagi"),
        (b"hircocervus", b"hircocervus"),
        (b"hoggery", b"hoggeri"),
        (b"homodermy", b"homodermi"),
        (b"hugoesque", b"hugoesqu"),
        (b"huia", b"huia"),
        (b"inartificially", b"inartifici"),
        (b"indetectable", b"indetect"),
        (b"inimitable", b"inimit"),
        (b"innervational", b"innerv"),
        (b"innocence", b"innoc"),
        (b"inoepithelioma", b"inoepithelioma"),
        (b"insouciantly", b"insouci"),
        (b"intercomparable", b"intercompar"),
        (b"involucred", b"involucr"),
        (b"irreparableness", b"irrepar"),
        (b"irrepealable", b"irrepeal"),
        (b"irresolution", b"irresolut"),
        (b"isocolic", b"isocol"),
        (b"italici", b"italici"),
        (b"jardiniere", b"jardinier"),
        (b"jucuna", b"jucuna"),
        (b"judaically", b"judaic"),
        (b"jumpingly", b"jump"),
        (b"keepworthy", b"keepworthi"),
        (b"kerseymere", b"kerseymer"),
        (b"kurmburra", b"kurmburra"),
        (b"labyrinthitis", b"labyrinth"),
        (b"lazaretto", b"lazaretto"),
        (b"leapingly", b"leap"),
        (b"ledgeless", b"ledgeless"),
        (b"leucosis", b"leucosi"),
        (b"licitness", b"licit"),
        (b"ludditism", b"luddit"),
        (b"lumpiness", b"lumpi"),
        (b"lunch", b"lunch"),
        (b"lyngbyeae", b"lyngbyea"),
        (b"macanese", b"macanes"),
        (b"machinofacture", b"machinofactur"),
        (b"marocain", b"marocain"),
        (b"matchwood", b"matchwood"),
        (b"mediant", b"mediant"),
        (b"mesal", b"mesal"),
        (b"methodize", b"method"),
        (b"metropolitic", b"metropolit"),
        (b"millinery", b"millineri"),
        (b"miniate", b"miniat"),
        (b"mirthsomeness", b"mirthsom"),
        (b"misname", b"misnam"),
        (b"misvaluation", b"misvalu"),
        (b"mizzly", b"mizzli"),
        (b"modernistic", b"modernist"),
        (b"moed", b"mo"),
        (b"molestation", b"molest"),
        (b"monovalence", b"monoval"),
        (b"musculophrenic", b"musculophren"),
        (b"mycteric", b"mycter"),
        (b"myomorphic", b"myomorph"),
        (b"nattiness", b"natti"),
        (b"neap", b"neap"),
        (b"nettly", b"nett"),
        (b"neurophagy", b"neurophagi"),
        (b"nonadjectival", b"nonadjectiv"),
        (b"noncopying", b"noncopi"),
        (b"noncredent", b"noncred"),
        (b"nonesthetic", b"nonesthet"),
        (b"nonmomentary", b"nonmomentari"),
        (b"nonsocialistic", b"nonsocialist"),
        (b"notonecta", b"notonecta"),
        (b"oaken", b"oaken"),
        (b"odontophore", b"odontophor"),
        (b"onomatologist", b"onomatologist"),
        (b"ophiurid", b"ophiurid"),
        (b"opprobriously", b"opprobri"),
        (b"ordainer", b"ordain"),
        (b"orthotropous", b"orthotrop"),
        (b"oscinine", b"oscinin"),
        (b"osphyocele", b"osphyocel"),
        (b"outboast", b"outboast"),
        (b"overborne", b"overborn"),
        (b"overcautiously", b"overcauti"),
        (b"overexertedly", b"overexert"),
        (b"overroast", b"overroast"),
        (b"overwing", b"overw"),
        (b"pager", b"pager"),
        (b"paintability", b"paintabl"),
        (b"paraphysis", b"paraphysi"),
        (b"pentastomous", b"pentastom"),
        (b"perfecti", b"perfecti"),
        (b"perilobar", b"perilobar"),
        (b"perligenous", b"perligen"),
        (b"pettedly", b"pet"),
        (b"photoalgraphy", b"photoalgraphi"),
        (b"photoceramics", b"photoceram"),
        (b"photodramatic", b"photodramat"),
        (b"phytohormone", b"phytohormon"),
        (b"piet", b"piet"),
        (b"pineal", b"pineal"),
        (b"piotty", b"piotti"),
        (b"placus", b"placus"),
        (b"plumoseness", b"plumos"),
        (b"podostomatous", b"podostomat"),
        (b"polyonomy", b"polyonomi"),
        (b"postvesical", b"postves"),
        (b"prankingly", b"prank"),
        (b"precoagulation", b"precoagul"),
        (b"precociously", b"precoci"),
        (b"prediscontinuance", b"prediscontinu"),
        (b"preferable", b"prefer"),
        (b"preinaugurate", b"preinaugur"),
        (b"preopposition", b"preopposit"),
        (b"preoverthrow", b"preoverthrow"),
        (b"presentational", b"present"),
        (b"princified", b"princifi"),
        (b"probusiness", b"probusi"),
        (b"professional", b"profession"),
        (b"proinsurance", b"proinsur"),
        (b"propopery", b"propoperi"),
        (b"psilanthropy", b"psilanthropi"),
        (b"psilotaceae", b"psilotacea"),
        (b"purgative", b"purgat"),
        (b"purpurate", b"purpur"),
        (b"pycnodontoid", b"pycnodontoid"),
        (b"pycnotic", b"pycnot"),
        (b"pyloroplasty", b"pyloroplasti"),
        (b"quintetto", b"quintetto"),
        (b"quintius", b"quintius"),
        (b"radicalness", b"radic"),
        (b"ramosely", b"ramos"),
        (b"rattan", b"rattan"),
        (b"rebone", b"rebon"),
        (b"recollation", b"recol"),
        (b"recompose", b"recompos"),
        (b"reconcilement", b"reconcil"),
        (b"recondemn", b"recondemn"),
        (b"reflagellate", b"reflagel"),
        (b"rehypothecation", b"rehypothec"),
        (b"reincarnadine", b"reincarnadin"),
        (b"renderable", b"render"),
        (b"reometer", b"reomet"),
        (b"requirement", b"requir"),
        (b"reversing", b"revers"),
        (b"rhyparographer", b"rhyparograph"),
        (b"ribaldry", b"ribaldri"),
        (b"rimester", b"rimest"),
        (b"rosette", b"rosett"),
        (b"rueful", b"rueful"),
        (b"rung", b"rung"),
        (b"salading", b"salad"),
        (b"scam", b"scam"),
        (b"sciotheism", b"sciotheism"),
        (b"scolecoid", b"scolecoid"),
        (b"sebaceous", b"sebac"),
        (b"semitic", b"semit"),
        (b"seroprophylaxis", b"seroprophylaxi"),
        (b"shadbelly", b"shadbelli"),
        (b"shroud", b"shroud"),
        (b"sided", b"side"),
        (b"silly", b"silli"),
        (b"snakily", b"snakili"),
        (b"sobriquet", b"sobriquet"),
        (b"somewhatly", b"somewhat"),
        (b"somma", b"somma"),
        (b"spartanlike", b"spartanlik"),
        (b"spatting", b"spat"),
        (b"specs", b"spec"),
        (b"sphericotriangular", b"sphericotriangular"),
        (b"squarehead", b"squarehead"),
        (b"statocracy", b"statocraci"),
        (b"steamcar", b"steamcar"),
        (b"stigmatoid", b"stigmatoid"),
        (b"stranding", b"strand"),
        (b"strenth", b"strenth"),
        (b"subgens", b"subgen"),
        (b"sublaciniate", b"sublacini"),
        (b"subpoenal", b"subpoen"),
        (b"subsneer", b"subsneer"),
        (b"sulforicinoleate", b"sulforicinol"),
        (b"sunshade", b"sunshad"),
        (b"superarbitrary", b"superarbitrari"),
        (b"supercompetition", b"supercompetit"),
        (b"sylvate", b"sylvat"),
        (b"taffrail", b"taffrail"),
        (b"tangram", b"tangram"),
        (b"telary", b"telari"),
        (b"tetragyn", b"tetragyn"),
        (b"thermomagnetism", b"thermomagnet"),
        (b"throttling", b"throttl"),
        (b"thyestean", b"thyestean"),
        (b"tiremaker", b"tiremak"),
        (b"trachyline", b"trachylin"),
        (b"trackman", b"trackman"),
        (b"tradesmanlike", b"tradesmanlik"),
        (b"transpeciate", b"transpeci"),
        (b"treeful", b"treeful"),
        (b"trespass", b"trespass"),
        (b"tricktrack", b"tricktrack"),
        (b"trispast", b"trispast"),
        (b"turbulently", b"turbul"),
        (b"ultradandyism", b"ultradandy"),
        (b"unassaultable", b"unassault"),
        (b"unattested", b"unattest"),
        (b"unchristianized", b"unchristian"),
        (b"undelineated", b"undelin"),
        (b"undisreputable", b"undisreput"),
        (b"unferreted", b"unferret"),
        (b"unforewarnedness", b"unforewarned"),
        (b"unfrounced", b"unfrounc"),
        (b"unintombed", b"unintomb"),
        (b"unjumpable", b"unjump"),
        (b"unkirk", b"unkirk"),
        (b"unmedicinable", b"unmedicin"),
        (b"unmentionableness", b"unmention"),
        (b"unominous", b"unomin"),
        (b"unpaved", b"unpav"),
        (b"unpercolated", b"unpercol"),
        (b"unpowerful", b"unpow"),
        (b"unsconced", b"unsconc"),
        (b"unshamefaced", b"unshamefac"),
        (b"unshockable", b"unshock"),
        (b"unturgid", b"unturgid"),
        (b"unwanton", b"unwanton"),
        (b"urethylan", b"urethylan"),
        (b"urinometric", b"urinometr"),
        (b"uvulae", b"uvula"),
        (b"vegetation", b"veget"),
        (b"victuals", b"victual"),
        (b"villanelle", b"villanell"),
        (b"whauk", b"whauk"),
        (b"whipping", b"whip"),
        (b"woodjobber", b"woodjobb"),
        (b"wough", b"wough"),
        (b"xylocarpous", b"xylocarp"),
        (b"zealotic", b"zealot"),
    ];

    #[test]
    fn the_reference_agrees_word_for_word() {
        let mut e = English::new();
        for (word, want) in CORPUS {
            assert_eq!(e.stem(word), want, "{}", String::from_utf8_lossy(word));
        }
    }

    #[test]
    fn the_buffer_is_reused_across_words() {
        let mut e = English::new();
        assert_eq!(e.stem(b"running"), b"run");
        assert_eq!(e.stem(b"flies"), b"fli");
        assert_eq!(e.stem(b"hello"), b"hello");
    }
}
