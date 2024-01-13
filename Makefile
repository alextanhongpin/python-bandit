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
