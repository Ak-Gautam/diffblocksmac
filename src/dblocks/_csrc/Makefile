# dblocks native C extension
# Builds libdblocks_native.dylib for macOS arm64

CC = clang
CFLAGS = -O3 -march=native -fPIC -Wall -Wextra
LDFLAGS = -dynamiclib

TARGET = libdblocks_native.dylib
SRC = dblocks_native.c
OBJ = $(SRC:.c=.o)

# Install into the Python package directory
INSTALL_DIR = ..

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

install: $(TARGET)
	cp $(TARGET) $(INSTALL_DIR)/$(TARGET)

clean:
	rm -f $(OBJ) $(TARGET)
