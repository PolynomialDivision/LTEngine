//! Small bounded in-memory LRU cache for translations. Disabled by default
//! because it keeps message text in memory.

use std::collections::HashMap;
use std::sync::Mutex;

/// Cache key: everything that affects the model output.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub source: String,
    pub target: String,
    pub format: String,
    pub text: String,
}

#[derive(Debug)]
struct Entry {
    value: String,
    last_used: u64,
}

/// Least-recently-used cache. Eviction is O(n), which is fine for the few
/// hundred entries this is meant for.
#[derive(Debug)]
pub struct TranslationCache {
    capacity: usize,
    inner: Mutex<(HashMap<CacheKey, Entry>, u64)>,
}

impl TranslationCache {
    /// Create a cache holding at most `capacity` translations (0 disables it).
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            inner: Mutex::new((HashMap::new(), 0)),
        }
    }

    /// Look up a translation.
    pub fn get(&self, key: &CacheKey) -> Option<String> {
        if self.capacity == 0 {
            return None;
        }
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (map, clock) = &mut *guard;
        *clock += 1;
        let entry = map.get_mut(key)?;
        entry.last_used = *clock;
        Some(entry.value.clone())
    }

    /// Store a translation, evicting the least recently used one if full.
    pub fn insert(&self, key: CacheKey, value: String) {
        if self.capacity == 0 {
            return;
        }
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (map, clock) = &mut *guard;
        *clock += 1;
        if map.len() >= self.capacity
            && !map.contains_key(&key)
            && let Some(oldest) = map
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| k.clone())
        {
            map.remove(&oldest);
        }
        map.insert(
            key,
            Entry {
                value,
                last_used: *clock,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(text: &str) -> CacheKey {
        CacheKey {
            source: "de".into(),
            target: "en".into(),
            format: "text".into(),
            text: text.into(),
        }
    }

    #[test]
    fn evicts_least_recently_used() {
        let cache = TranslationCache::new(2);
        cache.insert(key("a"), "A".into());
        cache.insert(key("b"), "B".into());
        assert_eq!(cache.get(&key("a")).as_deref(), Some("A"));
        cache.insert(key("c"), "C".into());
        assert!(cache.get(&key("b")).is_none());
        assert_eq!(cache.get(&key("a")).as_deref(), Some("A"));
        assert_eq!(cache.get(&key("c")).as_deref(), Some("C"));
    }

    #[test]
    fn disabled_cache_stores_nothing() {
        let cache = TranslationCache::new(0);
        cache.insert(key("a"), "A".into());
        assert!(cache.get(&key("a")).is_none());
    }
}
