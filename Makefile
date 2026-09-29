# VitaMind AI engine. One command starts both agents, each on its own port.
PY ?= python

.PHONY: dev mira lumina test test-mira test-lumina models clean-ports

dev:            ## both agents, restart-on-crash, status table
	$(PY) -m scripts.run_all

mira:           ## Mira only (:8101)
	$(PY) -m scripts.run_all --only mira

lumina:         ## Lumina only (:8102)
	$(PY) -m scripts.run_all --only lumina

dev-reload:     ## both agents with autoreload
	$(PY) -m scripts.run_all --reload

test: test-lumina test-mira

test-lumina:
	cd lumina && $(PY) -m pytest -q

test-mira:
	cd mira && $(PY) -m pytest -q

models:         ## rebuild datasets and retrain every Lumina model
	cd lumina && $(PY) -m data_prep.build_dialogue \
	  && $(PY) -m data_prep.build_safety \
	  && $(PY) -m data_prep.build_intent \
	  && $(PY) -m training.train_safety \
	  && $(PY) -m training.train_emotion \
	  && $(PY) -m training.train_intent \
	  && $(PY) -m training.registry
