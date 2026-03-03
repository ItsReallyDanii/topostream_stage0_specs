reproduce:
	python -m topostream.cli reproduce --config configs/default.yaml

validate:
	python -m topostream.cli validate --results-dir results/

test:
	python -m pytest tests/ -v

clean:
	rm -rf results/ figures/
