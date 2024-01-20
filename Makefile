vw := poetry run python -m vowpalwabbit

train:
	@$(vw) -d train.txt -f model.vw

test:
	@$(vw) -d test.txt -i model.vw -p predictions.txt

house:
	@$(vw) house_dataset
	@$(vw) house_dataset -l 10 -c --passes 25 --holdout_off -f house.model

kernel:
	python -m pip install ipykernel
	python -m ipykernel install --user

setup-formatter:
	poetry add jupyterlab-code-formatter
	poetry add black isort
	jupyter nbextension enable --py widgetsnbextension

convert_all:
	# jupytext doesn't preserve image.
	#@find . -name "*.ipynb" ! -path '*/.*' -exec poetry run jupytext --to md {} \;
	@find . -name "*.ipynb" ! -path '*/.*' -exec poetry run jupyter nbconvert --to markdown --output-dir=docs {} \;

# Similar to convert, but only convert the diff files.
convert:
	@poetry run jupyter nbconvert --to markdown --output-dir=docs $(shell git diff HEAD --name-only | grep .ipynb)


cover:
	@# The s prints the print output to stdout.
	poetry run coverage run -m pytest -s
	poetry run coverage report -m
	poetry run coverage html
	open htmlcov/index.html
