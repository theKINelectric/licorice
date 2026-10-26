//! Deterministic resolution of time expressions against an anchor date (SPEC §8.2).
//!
//! The extractor copies time words from the dialogue ("yesterday", "last Saturday",
//! "two weeks ago", "in May"); this module turns them into a date interval. The same
//! scanner finds time windows in questions (SPEC §9.2). Dates are proleptic Gregorian
//! days counted from 1970-01-01; there are no time zones at this level.

/// A calendar day, as days since 1970-01-01.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Day(pub i64);

impl Day {
    /// From a civil date; `None` if it doesn't exist.
    pub fn from_ymd(y: i64, m: u32, d: u32) -> Option<Day> {
        if !(1..=12).contains(&m) || d == 0 || d > days_in_month(y, m) {
            return None;
        }
        // Howard Hinnant's days_from_civil.
        let y = if m <= 2 { y - 1 } else { y };
        let era = y.div_euclid(400);
        let yoe = y - era * 400;
        let mp = (m as i64 + 9) % 12;
        let doy = (153 * mp + 2) / 5 + d as i64 - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        Some(Day(era * 146_097 + doe - 719_468))
    }

    pub fn ymd(self) -> (i64, u32, u32) {
        let z = self.0 + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
        let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
        let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
        (y, m, d)
    }

    /// 0 = Monday … 6 = Sunday.
    pub fn weekday(self) -> u32 {
        (self.0 + 3).rem_euclid(7) as u32
    }

    pub fn plus(self, days: i64) -> Day {
        Day(self.0 + days)
    }

    /// Milliseconds since the Unix epoch at the start of this day (UTC).
    pub fn start_ms(self) -> i64 {
        self.0 * 86_400_000
    }

    /// Milliseconds at the last millisecond of this day (UTC).
    pub fn end_ms(self) -> i64 {
        self.start_ms() + 86_400_000 - 1
    }

    pub fn from_ms(ms: i64) -> Day {
        Day(ms.div_euclid(86_400_000))
    }

    pub fn iso(self) -> String {
        let (y, m, d) = self.ymd();
        format!("{y:04}-{m:02}-{d:02}")
    }

    /// "7 May 2023".
    pub fn human(self) -> String {
        let (y, m, d) = self.ymd();
        format!("{d} {} {y}", MONTHS[m as usize - 1].0)
    }
}

pub fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
        2 => 28,
        _ => 0,
    }
}

/// How precise a resolved time is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Day,
    Week,
    Month,
    Season,
    Year,
}

impl Precision {
    pub fn as_str(self) -> &'static str {
        match self {
            Precision::Day => "day",
            Precision::Week => "week",
            Precision::Month => "month",
            Precision::Season => "season",
            Precision::Year => "year",
        }
    }

    /// The inverse of [`Precision::as_str`].
    pub fn parse(s: &str) -> Option<Precision> {
        [Precision::Day, Precision::Week, Precision::Month, Precision::Season, Precision::Year]
            .into_iter()
            .find(|p| p.as_str() == s)
    }
}

/// An inclusive interval of days.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Interval {
    pub from: Day,
    pub to: Day,
    pub precision: Precision,
}

impl Interval {
    fn day(d: Day) -> Interval {
        Interval { from: d, to: d, precision: Precision::Day }
    }

    fn month(y: i64, m: u32) -> Interval {
        let from = Day::from_ymd(y, m, 1).expect("valid month");
        Interval { from, to: from.plus(days_in_month(y, m) as i64 - 1), precision: Precision::Month }
    }

    fn year(y: i64) -> Interval {
        Interval {
            from: Day::from_ymd(y, 1, 1).expect("valid year"),
            to: Day::from_ymd(y, 12, 31).expect("valid year"),
            precision: Precision::Year,
        }
    }

    fn week_of(d: Day) -> Interval {
        let monday = d.plus(-(d.weekday() as i64));
        Interval { from: monday, to: monday.plus(6), precision: Precision::Week }
    }

    /// Renders the interval at its own precision, so a vague time never reads as an
    /// exact day: "7 May 2023", "week of 1 May 2023", "May 2023", "summer 2022",
    /// "winter 2022–2023", "2022", or else a day range ("6–7 May 2023").
    pub fn human(&self) -> String {
        let (fy, fm, fd) = self.from.ymd();
        let (ty, tm, td) = self.to.ymd();
        match self.precision {
            Precision::Week if self.to.0 - self.from.0 == 6 && self.from.weekday() == 0 => {
                return format!("week of {}", self.from.human());
            }
            Precision::Month if *self == Interval::month(fy, fm) => {
                return format!("{} {fy}", MONTHS[fm as usize - 1].0);
            }
            Precision::Season if fd == 1 && *self == season_interval(fy, fm) => match fm {
                3 => return format!("spring {fy}"),
                6 => return format!("summer {fy}"),
                9 => return format!("fall {fy}"),
                12 => return format!("winter {fy}–{}", fy + 1),
                _ => {}
            },
            Precision::Year if *self == Interval::year(fy) => return fy.to_string(),
            _ => {}
        }
        if self.to <= self.from {
            self.from.human()
        } else if (fy, fm) == (ty, tm) {
            format!("{fd}–{td} {} {fy}", MONTHS[fm as usize - 1].0)
        } else if fy == ty {
            format!("{fd} {} – {td} {} {fy}", MONTHS[fm as usize - 1].0, MONTHS[tm as usize - 1].0)
        } else {
            format!("{} – {}", self.from.human(), self.to.human())
        }
    }
}

const MONTHS: [(&str, &str); 12] = [
    ("January", "jan"),
    ("February", "feb"),
    ("March", "mar"),
    ("April", "apr"),
    ("May", "may"),
    ("June", "jun"),
    ("July", "jul"),
    ("August", "aug"),
    ("September", "sep"),
    ("October", "oct"),
    ("November", "nov"),
    ("December", "dec"),
];

const WEEKDAYS: [&str; 7] = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"];

fn month_number(word: &str) -> Option<u32> {
    let w = word.trim_end_matches('.');
    MONTHS
        .iter()
        .position(|(full, short)| w == full.to_lowercase() || w == *short || (w == "sept" && *short == "sep"))
        .map(|i| i as u32 + 1)
}

fn weekday_number(word: &str) -> Option<u32> {
    WEEKDAYS
        .iter()
        .position(|d| word == *d || (word.len() >= 3 && d.starts_with(word) && word.len() <= 4))
        .map(|i| i as u32)
}

/// A word that names a time only in context: a month ("may", "march", "june"), a
/// weekday abbreviation ("sun", "sat", "wed") or a four-digit year ("1984").
fn ambiguous_alone(word: &str) -> bool {
    month_number(word).is_some()
        || (weekday_number(word).is_some() && !WEEKDAYS.contains(&word))
        || (word.len() == 4 && word.bytes().all(|b| b.is_ascii_digit()))
}

fn count(word: &str) -> Option<i64> {
    if let Ok(n) = word.parse::<i64>() {
        return (0..=1000).contains(&n).then_some(n);
    }
    Some(match word {
        "a" | "an" | "one" => 1,
        "two" | "couple" => 2,
        "three" | "few" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        "eleven" => 11,
        "twelve" => 12,
        _ => return None,
    })
}

/// A count at the start of `w` ("3", "two", "a couple of", "a few"): its value and how
/// many words it used.
fn quantity(w: &[&str]) -> Option<(i64, usize)> {
    let at = |i: usize| w.get(i).copied().unwrap_or("");
    count(at(0))?;
    match at(1) {
        "couple" | "few" => Some((count(at(1))?, if at(2) == "of" { 3 } else { 2 })),
        _ => Some((count(at(0))?, 1)),
    }
}

/// The day, week, month or year `n` units (`unit`, singular or plural) from `anchor`.
fn offset(anchor: Day, unit: &str, n: i64) -> Option<Interval> {
    let (y, m, _) = anchor.ymd();
    match unit.trim_end_matches('s') {
        "day" => Some(Interval::day(anchor.plus(n))),
        "week" => Some(Interval::week_of(anchor.plus(7 * n))),
        "month" => {
            let (y, m) = add_months(y, m, n);
            Some(Interval::month(y, m))
        }
        "year" => Some(Interval::year(y + n)),
        _ => None,
    }
}

fn add_months(y: i64, m: u32, delta: i64) -> (i64, u32) {
    let idx = y * 12 + (m as i64 - 1) + delta;
    (idx.div_euclid(12), (idx.rem_euclid(12) + 1) as u32)
}

/// Seasons, northern hemisphere: (name, first month). Winter starts in December.
const SEASONS: [(&str, u32); 5] = [("spring", 3), ("summer", 6), ("fall", 9), ("autumn", 9), ("winter", 12)];

fn season_interval(year: i64, first_month: u32) -> Interval {
    let from = Day::from_ymd(year, first_month, 1).expect("valid season");
    let (ey, em) = add_months(year, first_month, 2);
    let to = Day::from_ymd(ey, em, days_in_month(ey, em)).expect("valid season");
    Interval { from, to, precision: Precision::Season }
}

/// Season `first_month` relative to `anchor`: `last` = the latest one completed before
/// it, `next` = the first one starting after it, `this` = the one containing it or else
/// the nearest ("this summer" said in May is the coming summer, not last year's).
fn season_relative(anchor: Day, first_month: u32, which: &str) -> Interval {
    let (y, _, _) = anchor.ymd();
    let mut candidates: Vec<Interval> = (y - 2..=y + 1).map(|yy| season_interval(yy, first_month)).collect();
    candidates.sort_by_key(|i| i.from);
    match which {
        "next" => *candidates.iter().find(|i| i.from > anchor).unwrap_or(&candidates[candidates.len() - 1]),
        "this" => *candidates
            .iter()
            .min_by_key(|i| if anchor < i.from { i.from.0 - anchor.0 } else { (anchor.0 - i.to.0).max(0) })
            .expect("four candidate seasons"),
        _ => *candidates.iter().rev().find(|i| i.to < anchor).unwrap_or(&candidates[0]),
    }
}

fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !(c.is_alphanumeric() || c == '/' || c == '-'))
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Resolves a time expression relative to `anchor` (the day it was said).
/// Returns `None` for vague expressions ("recently", "a while ago") or no time at all.
pub fn resolve(expression: &str, anchor: Day) -> Option<Interval> {
    let w = words(expression);
    // "in two weeks" needs its "in", which the filter below drops as filler.
    let raw: Vec<&str> = w.iter().map(String::as_str).collect();
    if raw.first() == Some(&"in") {
        if let Some((interval, _)) = match_at(&raw, anchor) {
            return Some(interval);
        }
    }
    let w: Vec<&str> = w
        .iter()
        .map(String::as_str)
        .filter(|x| !matches!(*x, "on" | "in" | "at" | "around" | "about" | "the" | "of" | "during" | "back"))
        .collect();
    match_at(&w, anchor).map(|(interval, _)| interval)
}

/// Finds the first time expression anywhere in `text` (a question, say), relative to
/// `anchor`. Returns the interval and the matched words.
pub fn find(text: &str, anchor: Day) -> Option<(Interval, String)> {
    let original: Vec<&str> =
        text.split(|c: char| !(c.is_alphanumeric() || c == '/' || c == '-')).filter(|w| !w.is_empty()).collect();
    let all = words(text);
    let w: Vec<&str> = all.iter().map(String::as_str).collect();
    for start in 0..w.len() {
        if let Some((interval, len)) = match_at(&w[start..], anchor) {
            // Alone, "may", "march", "sun", "wed" and "1984" are usually not times; take
            // them as such only after a preposition.
            let after_preposition = start > 0
                && matches!(
                    w[start - 1],
                    "in" | "on"
                        | "of"
                        | "during"
                        | "since"
                        | "until"
                        | "till"
                        | "by"
                        | "from"
                        | "around"
                        | "through"
                        | "early"
                        | "late"
                        | "mid"
                );
            if len == 1 && ambiguous_alone(w[start]) && !after_preposition {
                continue;
            }
            // Preserve explicit month spelling and following date/location adjuncts.
            // Lowercase "this may lead/be/..." is still a modal phrase.
            let explicit_may = original.get(start + 1).is_some_and(|w| matches!(*w, "May" | "MAY"));
            let date_adjunct = w.get(start + 2).is_some_and(|w| {
                matches!(
                    *w,
                    "in" | "on" | "at" | "for" | "with" | "during" | "before" | "after" | "from" | "through" | "until"
                )
            });
            if len == 2
                && w[start] == "this"
                && w[start + 1] == "may"
                && start + 2 < w.len()
                && !explicit_may
                && !date_adjunct
            {
                continue;
            }
            return Some((interval, w[start..start + len].join(" ")));
        }
    }
    None
}

/// Tries to match a time expression at the start of `w`; returns the interval and how
/// many words it used.
fn match_at(w: &[&str], anchor: Day) -> Option<(Interval, usize)> {
    let at = |i: usize| w.get(i).copied().unwrap_or("");
    let (ay, am, _) = anchor.ymd();
    match at(0) {
        "today" | "tonight" => return Some((Interval::day(anchor), 1)),
        "yesterday" => return Some((Interval::day(anchor.plus(-1)), 1)),
        "tomorrow" => return Some((Interval::day(anchor.plus(1)), 1)),
        "this" if matches!(at(1), "morning" | "afternoon" | "evening") => return Some((Interval::day(anchor), 2)),
        "last" if at(1) == "night" => return Some((Interval::day(anchor.plus(-1)), 2)),
        "day" if at(1) == "before" && at(2) == "yesterday" => return Some((Interval::day(anchor.plus(-2)), 3)),
        _ => {}
    }
    // ISO-like dates: 2023-05-07, 2023/05/07.
    if let Some(d) = parse_numeric_date(at(0)) {
        return Some((Interval::day(d), 1));
    }
    // "last/this/next week|weekend|month|year|<weekday>|<season>"
    if let which @ ("last" | "this" | "next" | "past" | "previous" | "coming") = at(0) {
        let which = match which {
            "past" | "previous" => "last",
            "coming" => "next",
            other => other,
        };
        let shift = match which {
            "last" => -1,
            "next" => 1,
            _ => 0,
        };
        match at(1) {
            "week" => return Some((Interval::week_of(anchor.plus(7 * shift)), 2)),
            "weekend" => {
                let saturday = anchor.plus(5 - anchor.weekday() as i64);
                let base = if anchor.weekday() >= 5 && which == "this" {
                    anchor.plus(-(anchor.weekday() as i64 - 5))
                } else {
                    saturday
                };
                let sat = match which {
                    "last" => base.plus(-7),
                    "next" => base.plus(7),
                    _ => base,
                };
                return Some((Interval { from: sat, to: sat.plus(1), precision: Precision::Week }, 2));
            }
            "month" => {
                let (y, m) = add_months(ay, am, shift);
                return Some((Interval::month(y, m), 2));
            }
            "year" => return Some((Interval::year(ay + shift), 2)),
            day if weekday_number(day).is_some() => {
                let target = weekday_number(day).unwrap() as i64;
                let today = anchor.weekday() as i64;
                let d = match which {
                    "next" => anchor.plus((target - today - 1).rem_euclid(7) + 1),
                    "this" => anchor.plus(target - today),
                    _ => anchor.plus(-((today - target - 1).rem_euclid(7) + 1)),
                };
                return Some((Interval::day(d), 2));
            }
            season if SEASONS.iter().any(|(n, _)| *n == season) => {
                let first = SEASONS.iter().find(|(n, _)| *n == season).unwrap().1;
                return Some((season_relative(anchor, first, which), 2));
            }
            // "this May" = May of this year; "last May" / "next May" = the nearest May
            // strictly before / after the current month.
            month if month_number(month).is_some() => {
                let m = month_number(month).unwrap();
                let y = match which {
                    "last" if m >= am => ay - 1,
                    "next" if m <= am => ay + 1,
                    _ => ay,
                };
                return Some((Interval::month(y, m), 2));
            }
            _ => {}
        }
    }
    // "<n> <unit>s ago" / "a couple of weeks ago" / "a few days ago" / "two weeks later"
    if let Some((n, used)) = quantity(w) {
        let sign = match at(used + 1) {
            "ago" => Some(-1),
            "later" | "from" => Some(1),
            _ => None,
        };
        if let Some(i) = sign.and_then(|sign| offset(anchor, at(used), sign * n)) {
            return Some((i, used + 2));
        }
    }
    // "in <n> <unit>s": "in two weeks", "in a few days"
    if at(0) == "in" {
        if let Some((n, used)) = quantity(&w[1..]) {
            if let Some(i) = offset(anchor, at(used + 1), n) {
                return Some((i, used + 2));
            }
        }
    }
    // "<season> [of] <year>": "summer 2022"; winter Y runs from December Y.
    if let Some(&(_, first)) = SEASONS.iter().find(|(n, _)| *n == at(0)) {
        let used = if at(1) == "of" { 2 } else { 1 };
        if let Some(y) = at(used).parse::<i64>().ok().filter(|y| (1900..=2100).contains(y)) {
            return Some((season_interval(y, first), used + 1));
        }
    }
    // A bare weekday: the most recent one before the anchor (past tense is the norm in recollections).
    if let Some(target) = weekday_number(at(0)) {
        let today = anchor.weekday() as i64;
        let back = (today - target as i64).rem_euclid(7);
        let back = if back == 0 { 7 } else { back };
        return Some((Interval::day(anchor.plus(-back)), 1));
    }
    // "<month> <day>[,] [year]" / "<month> [year]"
    if let Some(m) = month_number(at(0)) {
        let day =
            at(1).trim_end_matches(|c: char| c.is_alphabetic()).parse::<u32>().ok().filter(|d| (1..=31).contains(d));
        let (year_word, used) = if day.is_some() { (at(2), 2) } else { (at(1), 1) };
        let year = year_word.parse::<i64>().ok().filter(|y| (1900..=2100).contains(y));
        let used = used + year.is_some() as usize;
        let y = year.unwrap_or_else(|| infer_year(anchor, m));
        return match day {
            Some(d) => Day::from_ymd(y, m, d).map(|d| (Interval::day(d), used)),
            None => Some((Interval::month(y, m), used)),
        };
    }
    // "<day> <month> [year]"
    if let Some(d) =
        at(0).trim_end_matches(|c: char| c.is_alphabetic()).parse::<u32>().ok().filter(|d| (1..=31).contains(d))
    {
        if let Some(m) = month_number(at(1)) {
            let year = at(2).parse::<i64>().ok().filter(|y| (1900..=2100).contains(y));
            let y = year.unwrap_or_else(|| infer_year(anchor, m));
            return Day::from_ymd(y, m, d).map(|day| (Interval::day(day), 2 + year.is_some() as usize));
        }
    }
    // A bare year.
    if let Ok(y) = at(0).parse::<i64>() {
        if (1900..=2100).contains(&y) && at(0).len() == 4 {
            return Some((Interval::year(y), 1));
        }
    }
    None
}

/// A month named without a year means the latest such month not after the anchor.
fn infer_year(anchor: Day, month: u32) -> i64 {
    let (y, m, _) = anchor.ymd();
    if month <= m { y } else { y - 1 }
}

fn parse_numeric_date(word: &str) -> Option<Day> {
    let parts: Vec<&str> = word.split(['-', '/']).collect();
    if parts.len() != 3 || parts[0].len() != 4 {
        return None;
    }
    let y = parts[0].parse().ok()?;
    let m = parts[1].parse().ok()?;
    let d = parts[2].parse().ok()?;
    Day::from_ymd(y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn day(y: i64, m: u32, d: u32) -> Day {
        Day::from_ymd(y, m, d).unwrap()
    }

    // The anchor of LoCoMo conv-26 session 1: Monday 8 May 2023.
    fn anchor() -> Day {
        day(2023, 5, 8)
    }

    #[test]
    fn civil_round_trip_and_weekday() {
        for (y, m, d) in [(1970, 1, 1), (2000, 2, 29), (2023, 5, 8), (1999, 12, 31), (2100, 3, 1)] {
            assert_eq!(day(y, m, d).ymd(), (y, m, d));
        }
        assert_eq!(day(1970, 1, 1).0, 0);
        assert_eq!(day(2023, 5, 8).weekday(), 0, "8 May 2023 was a Monday");
        assert!(Day::from_ymd(2023, 2, 29).is_none());
        assert_eq!(day(2023, 5, 8).human(), "8 May 2023");
    }

    #[test]
    fn relative_days() {
        assert_eq!(resolve("yesterday", anchor()), Some(Interval::day(day(2023, 5, 7))));
        assert_eq!(resolve("today", anchor()), Some(Interval::day(anchor())));
        assert_eq!(resolve("last night", anchor()), Some(Interval::day(day(2023, 5, 7))));
        assert_eq!(resolve("the day before yesterday", anchor()), Some(Interval::day(day(2023, 5, 6))));
        assert_eq!(resolve("3 days ago", anchor()), Some(Interval::day(day(2023, 5, 5))));
        assert_eq!(resolve("a few days ago", anchor()), Some(Interval::day(day(2023, 5, 5))));
    }

    #[test]
    fn weekdays() {
        assert_eq!(resolve("last Saturday", anchor()), Some(Interval::day(day(2023, 5, 6))));
        assert_eq!(resolve("on Friday", anchor()), Some(Interval::day(day(2023, 5, 5))));
        assert_eq!(resolve("last Monday", anchor()), Some(Interval::day(day(2023, 5, 1))), "strictly before");
        assert_eq!(resolve("next Friday", anchor()), Some(Interval::day(day(2023, 5, 12))));
    }

    #[test]
    fn weeks_months_years_and_seasons() {
        let last_week = resolve("last week", anchor()).unwrap();
        assert_eq!((last_week.from, last_week.to), (day(2023, 5, 1), day(2023, 5, 7)));
        let two_weeks = resolve("two weeks ago", anchor()).unwrap();
        assert_eq!((two_weeks.from, two_weeks.precision), (day(2023, 4, 24), Precision::Week));
        let weekend = resolve("last weekend", anchor()).unwrap();
        assert_eq!((weekend.from, weekend.to), (day(2023, 5, 6), day(2023, 5, 7)));
        assert_eq!(resolve("last month", anchor()), Some(Interval::month(2023, 4)));
        assert_eq!(resolve("next month", anchor()), Some(Interval::month(2023, 6)));
        assert_eq!(resolve("last year", anchor()), Some(Interval::year(2022)));
        assert_eq!(resolve("a couple of months ago", anchor()), Some(Interval::month(2023, 3)));
        let summer = resolve("last summer", anchor()).unwrap();
        assert_eq!((summer.from, summer.to), (day(2022, 6, 1), day(2022, 8, 31)));
    }

    #[test]
    fn named_dates() {
        assert_eq!(resolve("on 7 May 2023", anchor()), Some(Interval::day(day(2023, 5, 7))));
        assert_eq!(resolve("May 7th", anchor()), Some(Interval::day(day(2023, 5, 7))));
        assert_eq!(resolve("in July", anchor()), Some(Interval::month(2022, 7)), "a later month means last year");
        assert_eq!(resolve("in March 2022", anchor()), Some(Interval::month(2022, 3)));
        assert_eq!(resolve("2022", anchor()), Some(Interval::year(2022)));
        assert_eq!(resolve("2023/05/20", anchor()), Some(Interval::day(day(2023, 5, 20))));
    }

    #[test]
    fn vague_or_absent_times_stay_unresolved() {
        for e in ["recently", "a while ago", "someday", "", "when I was young"] {
            assert_eq!(resolve(e, anchor()), None, "{e:?}");
        }
    }

    #[test]
    fn find_in_questions() {
        let (i, words) = find("What did Caroline do last weekend?", anchor()).unwrap();
        assert_eq!(words, "last weekend");
        assert_eq!(i.from, day(2023, 5, 6));
        let (i, _) = find("When in May 2023 did Melanie paint?", anchor()).unwrap();
        assert_eq!(i, Interval::month(2023, 5));
        assert!(find("What is Caroline's favourite book?", anchor()).is_none());
        assert!(find("What may Caroline decide to do?", anchor()).is_none(), "modal verb");
        assert_eq!(find("What did she do in May?", anchor()).unwrap().0, Interval::month(2023, 5));
    }

    #[test]
    fn intervals_render_at_their_precision() {
        let human = |e: &str| resolve(e, anchor()).unwrap().human();
        assert_eq!(human("yesterday"), "7 May 2023");
        assert_eq!(human("last week"), "week of 1 May 2023");
        assert_eq!(human("last weekend"), "6–7 May 2023");
        assert_eq!(human("last month"), "April 2023");
        assert_eq!(human("last summer"), "summer 2022");
        assert_eq!(human("this winter"), "winter 2022–2023");
        assert_eq!(human("last year"), "2022");
        let odd = Interval { from: day(2022, 12, 30), to: day(2023, 1, 2), precision: Precision::Day };
        assert_eq!(odd.human(), "30 December 2022 – 2 January 2023");
        for p in [Precision::Day, Precision::Week, Precision::Month, Precision::Season, Precision::Year] {
            assert_eq!(Precision::parse(p.as_str()), Some(p));
        }
    }

    #[test]
    fn ordinary_words_in_questions_are_not_dates() {
        for q in [
            "Does Melanie like the sun?",
            "When did they wed?",
            "Did she sat the exam?",
            "Has she read 1984?",
            "Did they march for equal rights?",
            "What did June say about painting?",
            "What did Caroline think this may lead to?",
        ] {
            assert_eq!(find(q, anchor()), None, "{q:?}");
        }
        assert_eq!(find("What did she do on Sat?", anchor()).unwrap().0, Interval::day(day(2023, 5, 6)));
        assert_eq!(find("Where did she live in 2022?", anchor()).unwrap().0, Interval::year(2022));
        assert_eq!(find("What did she paint this May?", anchor()).unwrap().0, Interval::month(2023, 5));
        assert_eq!(find("What did Mel do Monday?", anchor()).unwrap().0, Interval::day(day(2023, 5, 1)));
    }

    #[test]
    fn may_window_can_have_following_details_without_turning_modal_verbs_into_dates() {
        for q in [
            "What did Caroline paint this May in Oslo?",
            "What did Caroline paint this may for the exhibition?",
            "Where did she go this May with Melanie?",
        ] {
            assert_eq!(find(q, anchor()).map(|(i, _)| i), Some(Interval::month(2023, 5)), "{q}");
        }
        for q in [
            "What did Caroline think this may lead to?",
            "Does she think this may be helpful?",
            "What does this may mean?",
        ] {
            assert_eq!(find(q, anchor()), None, "{q}");
        }
    }

    #[test]
    fn this_season_is_the_current_or_nearest_one() {
        let span = |e: &str, at: Day| resolve(e, at).map(|i| (i.from, i.to));
        assert_eq!(span("this summer", anchor()), Some((day(2023, 6, 1), day(2023, 8, 31))), "said in May");
        assert_eq!(span("this winter", day(2023, 11, 15)), Some((day(2023, 12, 1), day(2024, 2, 29))));
        assert_eq!(span("this winter", day(2024, 1, 10)), Some((day(2023, 12, 1), day(2024, 2, 29))), "inside it");
        assert_eq!(span("last summer", anchor()), Some((day(2022, 6, 1), day(2022, 8, 31))));
        assert_eq!(span("next summer", anchor()), Some((day(2023, 6, 1), day(2023, 8, 31))));
    }

    #[test]
    fn relative_months_seasons_with_years_and_future_offsets() {
        assert_eq!(resolve("this May", anchor()), Some(Interval::month(2023, 5)));
        assert_eq!(resolve("last May", anchor()), Some(Interval::month(2022, 5)), "the current month is not last");
        assert_eq!(resolve("last March", anchor()), Some(Interval::month(2023, 3)));
        assert_eq!(resolve("next July", anchor()), Some(Interval::month(2023, 7)));
        assert_eq!(resolve("next May", anchor()), Some(Interval::month(2024, 5)));
        let summer = resolve("summer 2022", anchor()).unwrap();
        assert_eq!((summer.from, summer.to, summer.precision), (day(2022, 6, 1), day(2022, 8, 31), Precision::Season));
        assert_eq!(resolve("in the summer of 2022", anchor()), Some(summer));
        let winter = resolve("winter 2022", anchor()).unwrap();
        assert_eq!((winter.from, winter.to), (day(2022, 12, 1), day(2023, 2, 28)));
        assert_eq!(resolve("in two weeks", anchor()), Some(Interval::week_of(day(2023, 5, 22))));
        assert_eq!(resolve("in 3 days", anchor()), Some(Interval::day(day(2023, 5, 11))));
        assert_eq!(resolve("in a few days", anchor()), Some(Interval::day(day(2023, 5, 11))));
        assert_eq!(resolve("in a month", anchor()), Some(Interval::month(2023, 6)));
        assert_eq!(resolve("in a year", anchor()), Some(Interval::year(2024)));
        assert_eq!(resolve("in May", anchor()), Some(Interval::month(2023, 5)), "plain months still resolve");
        let (i, words) = find("What is she planning in two weeks?", anchor()).unwrap();
        assert_eq!((i, words.as_str()), (Interval::week_of(day(2023, 5, 22)), "in two weeks"));
    }
}
