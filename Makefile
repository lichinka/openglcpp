CXX      = g++
CXXFLAGS = -O0 -g
INCLUDE  = -I.
LIBRARY  = -lGL -lGLEW -lglfw -lm
OBJS     = glRenderer.o main.o
BIN      = keccak

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LIBRARY) -o $@ $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(LIBRARY) $(INCLUDE) -o $@ -c $<

clean:
	rm -rf $(BIN) $(OBJS)
