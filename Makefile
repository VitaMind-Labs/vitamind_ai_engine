# VitaMind AI engine. One command starts all three agents, each on its own port.
PY ?= python

.PHONY: dev mira lumina spark test test-mira test-lumina test-spark models clean-ports

dev:            ## all three agents, restart-on-crash, status table
	$(PY) -m scripts.run_all

mira:           ## Mira only (:8101)
	$(PY) -m scripts.run_all --only mira

lumina:         ## Lumina only (:8102)
	$(PY) -m scripts.run_all --only lumina

spark:          ## Spark only (:8103)
	$(PY) -m scripts.run_all --only spark

dev-reload:     ## all agents with autoreload
	$(PY) -m scripts.run_all --reload

test: test-lumina test-spark test-mira

test-spark:
	cd spark && $(PY) -m pytest -q

test-lumina:
	cd lumina_agent && $(PY) -m pytest -q

test-mira:
	cd mira && $(PY) -m pytest -q

models:         ## rebuild datasets and retrain every Lumina model
	cd lumina_agent && $(PY) -m data_prep.build_dialogue \
	  && $(PY) -m data_prep.build_safety \
	  && $(PY) -m data_prep.build_intent \
	  && $(PY) -m training.train_safety \
	  && $(PY) -m training.train_emotion \
	  && $(PY) -m training.train_intent \
	  && $(PY) -m training.registry
