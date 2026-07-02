CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_80

.PHONY: core2-search core2-search-cuda verify-6round plan-6round clean

core2_search_data.hpp: export_core2_cpp_data.py
	python export_core2_cpp_data.py

core2_trail_search: core2_trail_search.cpp core2_search_data.hpp
	$(CXX) $(CXXFLAGS) core2_trail_search.cpp -o core2_trail_search

core2_trail_search_cuda: core2_trail_search_cuda.cu core2_search_data.hpp
	$(NVCC) $(NVCCFLAGS) core2_trail_search_cuda.cu -o core2_trail_search_cuda

core3_search_data.hpp: export_core3_cpp_data.py
	python export_core3_cpp_data.py

core3_trail_search: core3_trail_search.cpp core3_search_data.hpp
	$(CXX) $(CXXFLAGS) core3_trail_search.cpp -o core3_trail_search

core3_trail_search_cuda: core3_trail_search_cuda.cu core3_search_data.hpp
	$(NVCC) $(NVCCFLAGS) core3_trail_search_cuda.cu -o core3_trail_search_cuda

core2-search: core2_trail_search

core2-search-cuda: core2_trail_search_cuda

core3-search: core3_trail_search

core3-search-cuda: core3_trail_search_cuda

verify-6round:
	python verify_paper_6round_collision.py

plan-6round:
	python reproduce_core3_connector.py

search-core3-first-two:
	python search_core3_first_two_parallel.py

repair-core3-first-two:
	python search_core3_first_two_repair.py

search-core3-beta:
	python search_core3_beta_pairs.py

search-core3-pair:
	python search_core3_with_beta_pair.py

search-core3-row-reorder:
	python search_core3_row_reorder.py --from-reorder-best

repair-core3-pair:
	python repair_core3_with_beta_pair.py --from-search-best

clean:
	rm -f core2_trail_search core2_trail_search_cuda core3_trail_search core3_trail_search_cuda
