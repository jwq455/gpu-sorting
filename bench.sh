#!/bin/bas

tmp=tmp.txt
benchs=benchs.txt

rm "$benchs"
Qs=(21, 22, 23)
lgHs=(7, 8, 9)
Ns=(100, 100000, 1000000000)
Bs=(128, 256, 512)

for B in ${Bs[@]}; do
	for lgH in ${lgHs[@]}; do
		for Q in ${Qs[@]}; do
			for N in ${Ns[@]}; do
				make clean
				echo "B="$B" Q="$Q" lgH="$lgH" N="$N"" >> "$benchs"
				make B="$B" Q="$Q" lgH="$lgH" N="$N" >> "$tmp"
				t_cup=$(grep "CUB Sorting" "$tmp"| sed -E 's/.*runs in: ([0-9.]+) us,.*/\1/')
				t_rad=$(grep "Radix Sorting" "$tmp"| sed -E 's/.*runs in: ([0-9.]+) us,.*/\1/')
				k_cup=$(grep "CUB Sorting" "$tmp"| sed -E 's/.*Sorted keys per second: ([0-9.]+).*/\1/')
				k_rad=$(grep "Radix Sorting" "$tmp" | sed -E 's/.*Sorted keys per second: ([0-9.]+).*/\1/')
				fut_time=$(grep "data.in:" "$tmp" | sed -E 's/.*data.in: ([0-9.]+).*/\1/')
				#valid=$(grep "INVALID" "$tmp")
				echo "Run time Futhark: "$fut_time"" >> "$benchs"
				echo "-------------------------------------------" >> "$benchs"
				echo "Run time CUP sort: "$t_cup"" >> "$benchs"
				echo "Nr of keys sorted pr second: "$k_cup"" >> "$benchs"
				echo "-------------------------------------------" >> "$benchs"
				echo "Run time Radix sort: " $t_rad"" >> "$benchs"
				echo "Nr of keys sorted pr second: "$k_rad"" >> "$benchs"
				#if [[! -n "$valid"]]; then
   				#	 echo "INVLAID RESULT"
				#fi 
				echo "==========================================" >> "$benchs" 
				rm "$tmp"
			done
		done
	done 
done
