all: build validate

build:
	python release-preparation.py

validate:
	java -jar validator.jar -r /car-data/gb/vphas/psfcat/eso -m create
