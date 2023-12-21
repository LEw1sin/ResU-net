export FLASK_APP = app.py

dev:
	flask run -p 7777

test_predict:
	python3 predict2.py

.PHONY:

clear_test:
	rmdir ./results/testForPath

clear_all:
	rmdir ./results/*
