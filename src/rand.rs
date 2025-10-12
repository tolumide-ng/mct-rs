use getrandom::getrandom;

pub fn genrand(min: usize, max: usize) -> usize {
    assert!(
        min < max,
        "min must be less than max. min={min} -> max={max}"
    );
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
