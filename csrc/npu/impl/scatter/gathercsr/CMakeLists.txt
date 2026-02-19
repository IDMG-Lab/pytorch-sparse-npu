cmake_minimum_required(VERSION 3.16)

# Register this operator directory so the top-level host/kernel build can
# deterministically collect sources from:
#   gathercsr/op_host/*.cpp
#   gathercsr/op_kernel/*
#
# This makes per-operator CMakeLists.txt *functional* (not just comments), and
# keeps the repository layout clean: each operator owns its host/kernel sources.

register_operator(gathercsr)
