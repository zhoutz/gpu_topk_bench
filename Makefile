.PHONY: conf
conf:
	until cmake -B build; do\
		sleep 1;\
	done

.PHONY: build
build:
	cmake --build build --config Release -j8

result_file_name := main_result
.PHONY: run
run:build
	./build/main_bench --csv ./build/$(result_file_name).csv                   \
	  --json ./build/$(result_file_name).json                                  \
	  --md ./build/$(result_file_name).md --min-time 1

.PHONY: clean
clean:
	rm -rf build test

.PHONY: test
test:
	nvcc test.cu -o test -run

.PHONY: plot
plot:
	python3 ./plot_bench.py ./build/$(result_file_name).csv
