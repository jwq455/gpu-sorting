import "lib/github.com/diku-dk/sorts/radix_sort"
-- 32-bit keys
-- ==
-- 
-- random input { [100000000]u32 }

let main(xs: []u32) : []u32 =
    radix_sort_int u32.num_bits u32.get_bit xs
