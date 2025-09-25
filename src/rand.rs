use getrandom::getrandom;

pub(crate) fn genrand(min: usize, max: usize) -> usize {
    assert!(min < max, "min must be less than max");
    let range = max - min;
    let mut buf = [0u8; std::mem::size_of::<usize>()];

    loop {
        getrandom(&mut buf).expect("random failed");
        let value = usize::from_ne_bytes(buf);
        let max_usable = usize::MAX - usize::MAX % range;
        if value < max_usable {
            return min + (value % range);
        }
        // else: retry
    }
}

/// TODO: Eventually replace with internally generated sequential ID (usize)
pub(crate) fn get_id() -> u64 {
    let mut buf = [0u8; 8];
    getrandom(&mut buf).unwrap();
    u64::from_le_bytes(buf)
}
