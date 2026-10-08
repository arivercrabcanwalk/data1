#!/usr/bin/env bash
# USER-EXECUTED ONLY. Does not start timers, send Feishu, or touch /dev/vdb or /data.
set -euo pipefail
DISK=/dev/vdc
TARGET=/srv/longtou-yimai-data
APP=/home/ubuntu/longtou-yimai
EXPECTED_BYTES=21474836480
if [[ ! -t 0 ]]; then echo "Interactive SSH required. Run: ssh -t lighthouse /home/ubuntu/longtou-yimai/initialize_new_disk.sh"; exit 60; fi
if [[ "$(id -un)" != ubuntu ]]; then echo "Must run as ubuntu"; exit 61; fi
if [[ ! -b "$DISK" || "$(sudo blockdev --getsize64 "$DISK")" != "$EXPECTED_BYTES" ]]; then echo "Disk /dev/vdc is not the expected NEW 20-GiB device"; exit 62; fi
if [[ -n "$(findmnt -n -S "$DISK" -o TARGET || true)" ]]; then echo "Disk already mounted; abort"; exit 63; fi
if [[ -n "$(sudo wipefs -n "$DISK" | sed '/^[[:space:]]*$/d')" ]]; then echo "The disk has signatures; abort rather than overwrite"; exit 64; fi
if sudo blkid -p "$DISK" >/dev/null 2>&1; then echo "Existing filesystem detected; abort"; exit 65; fi
if [[ -e "$TARGET" ]]; then echo "Dedicated mountpoint already exists; abort"; exit 66; fi
if [[ -e /dev/vdb ]]; then echo "Old device reappeared; request a fresh audit before proceeding"; exit 67; fi
echo "VERIFIED: /dev/vdc is the NEW 20-GiB blank volume with no filesystem signatures."
echo "This will create ext4 ONLY on /dev/vdc; old /data, SSH keys, business services remain untouched."
echo "Type FORMAT_NEW_EMPTY_VDC to proceed:"
read -r CONFIRM
if [[ "$CONFIRM" != FORMAT_NEW_EMPTY_VDC ]]; then echo "Cancelled, no disk changes made"; exit 68; fi
STAMP="$(date +%Y%m%dT%H%M%S)"
BACKUP="$APP/backups/user-format-vdc-$STAMP"
umask 077
mkdir -p "$BACKUP"
sudo cp -p /etc/fstab "$BACKUP/fstab.before"
lsblk -f "$DISK" > "$BACKUP/device.before"
sudo mkfs.ext4 -F -L longtou_yimai -m 1 "$DISK" > "$BACKUP/mkfs.log" 2>&1
UUID="$(sudo blkid -s UUID -o value "$DISK")"
[[ -n "$UUID" ]] || { echo "Filesystem created but UUID unavailable; stop"; exit 69; }
sudo install -d -m 755 "$TARGET"
if grep -qE '[[:space:]]/srv/longtou-yimai-data[[:space:]]' /etc/fstab; then echo "Duplicate fstab target, manual review required"; exit 70; fi
printf 'UUID=%s %s ext4 defaults,nofail,noatime,x-systemd.device-timeout=10s 0 2\n' "$UUID" "$TARGET" | sudo tee -a /etc/fstab >/dev/null
sudo mount "$TARGET"
sudo install -d -o ubuntu -g ubuntu -m 700 "$TARGET/state" "$TARGET/state/daily" "$TARGET/state/events" "$TARGET/state/signals" "$TARGET/state/delivery" "$TARGET/health" "$TARGET/archive"
python3 - <<'PY'
from pathlib import Path
import hashlib,os
target=Path('/srv/longtou-yimai-data')
assert os.path.ismount(str(target))
assert target.stat().st_dev!=Path('/').stat().st_dev
p=target/'health'/'first-roundtrip.bin'
payload=os.urandom(1048576)
with p.open('xb') as fd:fd.write(payload);fd.flush();os.fsync(fd.fileno())
assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(payload).digest()
p.unlink()
print('NEW_DISK_MOUNT_AND_FSYNC_ROUNDTRIP_OK')
PY
findmnt "$TARGET" -o SOURCE,TARGET,FSTYPE,OPTIONS
df -h "$TARGET"
echo "VDC_NEW_VOLUME_READY. Stop here; the assistant still needs to validate/activate the strategy."