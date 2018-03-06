# ======== COMPILER ========

CC 	= gcc
OPT	= -std=c99 -pedantic -Wall -O3 -Werror

# ======== LINKS ========

PROGDIR	= ..
SSTDIR	= $(PROGDIR)/cmb_sst_ksw
SSTLIB  = $(SSTDIR)/shared
SSTLIBNM= fisher
SSTSRC	= $(SSTDIR)/src
SSTBIN	= $(SSTDIR)/bin
SSTOBJ	= $(SSTSRC)
SSTINC	= $(SSTDIR)/include

WIGDIR		= $(PROGDIR)/wigxjpf-1.7
WIGINC		= $(WIGDIR)/inc/
WIGLIB		= $(WIGDIR)/lib/
WIGLIBNM    	= wigxjpf

# ======== SOURCE LOCATIONS ========

vpath %.c $(SSTSRC)
vpath %.h $(SSTSRC)

# ======== FFFLAGS ========

FFLAGS  = -I$(WIGINC) -I$(SSTINC)
FFLAGS	+= -fPIC

# ======== LDFLAGS ========

#LDFLAGS = -L$(SSTLIB) -l$(SSTLIBNM) -L$(WIGLIB) -l$(WIGLIBNM) -lm
LDFLAGS = -L$(WIGLIB) -l$(WIGLIBNM) -lm

# ======== OBJECT FILES TO MAKE ========

SSTOBJS = $(SSTOBJ)/sst_fisher.o

SSTHEADERS = sst_fisher.h

# ======== MAKE RULES ========

#$(SSTOBJ)/%.o: %.c $(SSTHEADERS)
#	        $(CC) $(OPT) $(FFLAGS) -c $< -o $@

#fisher : $(SSTOBJS) 
#	        $(CC) $(OPT) $(FFLAGS) $(LDFLAGS) -c $< -o $@
#$(SSTOBJS) : $(SSTOBJ)/sst_fisher.c ##works
#		$(CC) $(OPT)  $< $(LDFLAGS) $(FFLAGS) -o $(SSTBIN)/fisher



$(SSTLIB)/lib$(SSTLIBNM).so : $(SSTOBJ)/sst_fisher.o
		$(CC) $(OPT) $(FFLAGS) -shared  -o $(SSTLIB)/lib$(SSTLIBNM).so $(SSTOBJ)/sst_fisher.o  $(LDFLAGS) 

$(SSTOBJ)/sst_fisher.o : $(SSTOBJ)/sst_fisher.c $(SSTHEADERS)
		$(CC) $(OPT) $(FFLAGS) -c -o $@ $< $(LDFLAGS)  

.PHONY: test
test: $(SSTBIN)/fisher
$(SSTBIN)/fisher : $(SSTOBJ)/sst_fisher.o $(SSTLIB)/lib$(SSTLIBNM).so $(SSTHEADERS)
		$(CC) $(OPT) $(FFLAGS) $(LDFLAGS) -L$(SSTLIB) -l$(SSTLIBNM) -o $(SSTBIN)/fisher $(SSTOBJ)/sst_test_fisher.c  '-Wl,-rpath,$$ORIGIN/../shared'

#$(SSHTBIN)/ssht_test: $(SSHTOBJ)/ssht_test.o $(SSHTLIB)/lib$(SSHTLIBNM).a
#        $(CC) $(OPT) $< -o $(SSHTBIN)/ssht_test $(LDFLAGS)

#.PHONY: default
#default: shared

#.PHONY: shared
#shared: $(SSTLIB)/lib$(SSTLIBNM).so
#$(SSTLIB)/lib$(SSTLIBNM).so: $(SSTOBJS)
#	        $(CC) $(OPT) $(FFLAGS) -shared $(SSTLIB)/lib$(SSTLIBNM).so $(SSTOBJS)

.PHONY: clean
clean:
	rm -f $(SSTOBJ)/*.o
	rm -f $(SSTLIB)/lib$(SSTLIBNM).so
	rm -f $(SSTOBJMAT)/*.o
	rm -f $(SSTBIN)/fisher