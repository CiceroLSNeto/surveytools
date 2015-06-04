all: build validate

build:
	python release-preparation.py

validate:
	java -jar validator.jar -r /home/gb/tmp/eso -m create
