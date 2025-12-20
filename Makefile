# Compiler
CXX = g++

# Flags
CXXFLAGS = -std=c++11 -g -Iinclude

# Target executable
TARGET = test.a

# Find all .cpp files
SRCS = $(wildcard src/*.cpp) $(wildcard test/*.cpp)

# Convert .cpp → .o
OBJS = $(SRCS:.cpp=.o)

# Default: build & run
all: clean $(TARGET) run

# Link objects to produce executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile .cpp → .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run the program
run: $(TARGET)
	./$(TARGET)

# Cleanup
clean:
	rm -f $(OBJS) $(TARGET)
debug:
	gdb $(TARGET)