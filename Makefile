CXX=g++
CXXFLAGS= -g -Wall

ifeq ($(OS), Windows_NT)
	RM=cmd /C del
	ReplaceSeperator = $(subst /,\,$1)
else
	ifeq ($(shell uname), Linux)
		RM = rm -f
		ReplaceSeperator = $1
	endif
endif

SRCDIR=src
OBJDIR=bin

SRCS=$(SRCDIR)/Layer.cpp $(SRCDIR)/NeuralNetwork.cpp $(SRCDIR)/Neuron.cpp
OBJS=$(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))

all: $(OBJS) ;

clean:
	$(RM) $(call ReplaceSeperator, $(OBJS))

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJS): | $(OBJDIR)

$(OBJDIR):
	mkdir $(OBJDIR)