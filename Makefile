.PHONY: conf
conf:
	cmake -B build

.PHONY: build
build:
	cmake --build build --config Release -j8

result_file_name:=main_result0
.PHONY: run
run:build
	./build/main_bench --csv ./build/$(result_file_name).csv                   \
	  --json ./build/$(result_file_name).json                                \
	  --md ./build/$(result_file_name).md --min-time 1

.PHONY: clean
clean:
	rm -rf build

.PHONY: test
test:
	nvcc test.cu -o test -run

.PHONY: cmp
cmp:
	./nvbench/scripts/nvbench_compare.py ./build/main_result1.json ./build/main_result.json
