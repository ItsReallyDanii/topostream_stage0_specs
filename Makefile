reproduce:
	python -m topostream.cli reproduce --config configs/default.yaml

validate:
	python -m topostream.cli validate --results-dir results/

test:
	python -m pytest tests/ -v

benchmark-check:
	python benchmarks/stage1_xy_single_sweep/run_benchmark.py --check

benchmark-regenerate:
	python benchmarks/stage1_xy_single_sweep/run_benchmark.py --regenerate

clean:
	rm -rf results/ figures/ benchmarks/stage1_xy_single_sweep/output/
