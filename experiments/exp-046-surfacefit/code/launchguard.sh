#!/bin/bash
# The desk's launch guard for RESULTS section 47.17's screen.
#
# Clause (b), ruled 2026-09-09 after the original swap clause blocked a launch it should not
# have.  BOTH must pass:
#
#   1. kern.memorystatus_vm_pressure_level <= 2   (normal or warn; critical blocks)
#   2. free + inactive + purgeable >= FREE_MIN_GB
#   3. free + inactive + purgeable MINUS this launch's own planned footprint >= FREE_MIN_GB
#
# Exit 0 = clear to launch.  Exit 1 = BLOCKED; report and hold, never launch into it.
#
# WHY THE SWAP CLAUSE WAS REMOVED.  macOS swap is STICKY: pages written during other desks'
# jobs stay counted after those jobs exit, so `vm.swapusage free` is a high-water mark, not
# current pressure.  Swap used here is structurally ~8 GB from work that is not ours (iOS
# 19.7 GB, node 9.1, Chrome 9.1) and does not clear by waiting.  At the GO the swap clause
# blocked at 1.14 GB free while the kernel reported pressure NORMAL and no python job was
# alive.  The C2-cat precedent says the same from the other side: 4 GB RSS held for 78
# minutes under that identical swap state, never killed.
#
# PAGE SIZE, and a correction.  vm_stat on this machine reports a 16384-byte page (Apple
# Silicon), not 4096.  Every RAM figure this worker reported on 2026-09-09 was computed at
# 4096 and is therefore 4x UNDERSTATED -- the "3.27 GB" quoted to the desk when clause (b)
# was chosen is really 12.78 GB, on a 48 GB machine.  The page size is read from vm_stat
# here rather than assumed, so the reading cannot drift again.  The desk's >= 3 GB was
# chosen against the understated figure; FREE_MIN_GB is set to 3 GB as ruled and the true
# reading is printed beside it so the number can be re-set on correct arithmetic.
#
# Thresholds are the desk's.  A worker does not relax them at launch time to get a run going.

FREE_MIN_GB=3.0     # desk ruling 2026-09-09; see the page-size note above

# THE BUDGET CLAUSE, ruled 2026-09-09 (fourth reading), replacing a FLAT "no python job over
# 2 GB RSS".  The flat clause blocked the v23 dry run while queue's job held 4.09 GB and
# free+inactive stood at 13.25 GB -- it refused a launch on headroom that plainly existed,
# because it was written for the 1 GB swap-free regime and reads another desk's job as a
# hazard rather than as a subtraction.  The gate is now what actually matters: what is LEFT
# after this launch takes its share.
#
#   free+inactive+purgeable  -  PLAN_GB  >=  FREE_MIN_GB,   pressure <= 2
#
# Evidence the desk cited both ways: C2-cat ran 78 minutes at 4 GB RSS beside nothing, and
# the K = 10 extended bag ran 41 minutes at a 2.7 GB peak beside queue's 0.9 GB arm. Both sit
# inside this budget; neither was ever killed.
#
# PLAN_GB IS THE CALLER'S HONEST ESTIMATE and the guard cannot check it.  Pass the measured
# peak of the nearest comparable job plus margin -- for a serve on this panel that is 3.0,
# the bag's 2.7 GB peak rounded up.  Understating it is how this clause gets defeated.
PLAN_GB="${1:-3.0}"
RSS_RECORD_ONLY=1

# PRESSURE THRESHOLD, ruled 2026-09-09 (third reading).  The guard blocked a second time at
# level 2 sustained across 30 s of sampling while free+inactive held 12.7-13.1 GB on a 48 GB
# machine with NO python job alive (largest: ChatGPT 1.3 GB, Chrome 1.19, Xcode 0.55).  On this
# box level 2 tracks the compressor and the structurally full swap belonging to other desks --
# so clause (b) had inherited the very defect it was adopted to escape, and on that reasoning it
# would never clear.  Ruled: accept <= 2, block critical.  The live headroom is free+inactive,
# and the C2-cat precedent ran 78 minutes at 4 GB RSS under exactly this regime.
# NOTE the platform constants are 1 normal / 2 warn / 4 CRITICAL -- there is no 3 -- so `<= 2`
# is what blocks critical, and the desk's "level 3 blocks" is honoured by this test.
PRESS_MAX=2
press=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 1)
read -r free_gb ps_bytes <<<"$(vm_stat | awk '
  /page size of/{gsub(/[^0-9]/,"",$8); ps=$8}
  /Pages free/{f=$3} /Pages inactive/{i=$3} /Pages purgeable/{p=$3}
  END{gsub(/\./,"",f); gsub(/\./,"",i); gsub(/\./,"",p);
      printf "%.2f %d", (f+i+p)*ps/1e9, ps}')"
biggest=$(ps -Ao rss,pid,comm | grep -i python | sort -rn | head -1)
big_gb=$(echo "$biggest" | awk '{printf "%.2f", $1/1048576}'); [ -z "$big_gb" ] && big_gb=0.00
big_pid=$(echo "$biggest" | awk '{print $2}')

ok=0
printf 'pressure    %6s      (need <= %s; 1=normal 2=warn 4=critical)\n' "$press" "$PRESS_MAX"
printf 'free+inact  %6.2f GB   (need >= %.1f, page size %s)\n' "$free_gb" "$FREE_MIN_GB" "$ps_bytes"
printf 'largest py  %6.2f GB   pid %s (RECORDED ONLY -- subtracted as headroom, not a gate)\n' \
  "$big_gb" "${big_pid:-none}"
printf 'this launch %6.2f GB   planned footprint (caller-declared)\n' "$PLAN_GB"
printf 'headroom    %6.2f GB   after this launch (need >= %.1f)\n' \
  "$(awk -v f="$free_gb" -v p="$PLAN_GB" 'BEGIN{printf "%.2f", f-p}')" "$FREE_MIN_GB"
printf 'swap free   %6.2f GB   (RECORDED ONLY -- sticky high-water mark, not a gate)\n' \
  "$(sysctl -n vm.swapusage | sed -E 's/.*free = ([0-9.]+)M.*/\1/' | awk '{print $1/1024}')"

[ "$press" -le "$PRESS_MAX" ] 2>/dev/null || { echo "BLOCKED: kernel memory pressure (critical)"; ok=1; }
# clause 2 is SUBSUMED by the budget above (free-plan>=3 implies free>=3 for any plan >= 0)
# and is kept only so a zero-footprint caller still meets the desk's original floor
awk -v v="$free_gb" -v t="$FREE_MIN_GB" 'BEGIN{exit !(v>=t)}' || { echo "BLOCKED: free memory"; ok=1; }
awk -v f="$free_gb" -v p="$PLAN_GB" -v t="$FREE_MIN_GB" 'BEGIN{exit !(f-p>=t)}' || {
  echo "BLOCKED: budget -- free+inactive minus this launch leaves under ${FREE_MIN_GB} GB"; ok=1; }

if [ "$ok" -eq 0 ]; then echo "CLEAR TO LAUNCH"; else echo "HOLD -- report to the desk, do not launch"; fi
exit "$ok"
