CACHE := 'data/xy_cache.npz'

run:
	python run.py

clean:
	-rm $(CACHE)
