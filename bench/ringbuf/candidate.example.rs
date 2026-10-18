//! Reference (correct) implementation for bench/ringbuf.

pub struct RingBuf {
    cap: usize,
    buf: Vec<i64>,
    head: usize, // index of the front element
    len: usize,
}

pub fn ring_new(capacity: usize) -> RingBuf {
    let cap = capacity.max(1);
    RingBuf {
        cap,
        buf: vec![0; cap],
        head: 0,
        len: 0,
    }
}

pub fn ring_push(rb: &mut RingBuf, v: i64) {
    if rb.len < rb.cap {
        rb.buf[(rb.head + rb.len) % rb.cap] = v;
        rb.len += 1;
    } else {
        rb.buf[rb.head] = v; // overwrite the OLDEST slot
        rb.head = (rb.head + 1) % rb.cap;
    }
}

pub fn ring_pop(rb: &mut RingBuf) -> Option<i64> {
    if rb.len == 0 {
        return None;
    }
    let v = rb.buf[rb.head];
    rb.head = (rb.head + 1) % rb.cap;
    rb.len -= 1;
    Some(v)
}
