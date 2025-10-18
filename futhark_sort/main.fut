import "lib/github.com/diku-dk/sorts/radix_sort"
-- 32-bit keys
-- ==
-- entry: radix_sort_i32
-- random input { [100000]i32 }
-- random input { [1000000]i32 }
-- random input { [10000000]i32 }
-- random input { [100000000]i32 }

entry radix_sort_i32 = radix_sort 32 i32.get_bit