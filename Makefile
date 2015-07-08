all: build
	$(MAKE) -C build
	$(MAKE) -C python

build:
	mkdir build
	cd build && cmake ..
