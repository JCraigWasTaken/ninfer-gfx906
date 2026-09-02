# TP2 transport probes (gfx906 port). The donor built these standalone with
# nvcc; here they are ordinary HIP executables so they ride the same compat
# include tree and gfx906 flags as the rest of the tree. Run with both cards
# visible (HIP_VISIBLE_DEVICES=0,1); each exits 77 with fewer than two devices.
foreach(probe IN ITEMS p2p_probe transport_probe capture_probe)
  add_executable(ninfer_tp2_${probe} ${probe}.cu)
  ninfer_internal_includes(ninfer_tp2_${probe})
  ninfer_cuda_archive(ninfer_tp2_${probe})
  target_link_libraries(ninfer_tp2_${probe} PRIVATE ninfer_core)
endforeach()
