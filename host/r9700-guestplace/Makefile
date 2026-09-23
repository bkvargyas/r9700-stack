obj-m += r9700_guestplace.o
KDIR ?= /lib/modules/$(shell uname -r)/build
# the guest kernel (xanmod) is clang-built; pin the versioned toolchain here because dkms
# passes CC=clang, which Debian only ships as clang-<ver>
LLVM_VER ?= -21
KFLAGS = LLVM=$(LLVM_VER) CC=clang$(LLVM_VER) LD=ld.lld$(LLVM_VER)
all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) $(KFLAGS) modules
clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) $(KFLAGS) clean
