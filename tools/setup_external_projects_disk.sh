#!/usr/bin/env bash
set -euo pipefail

# One-time setup:
# - Repartition /dev/sdb as: single ext4 partition (full disk)
# - Persist mount in /etc/fstab
# - Migrate /home/warner/_projects to ext4 mount
# - Replace /home/warner/_projects with symlink to mounted ext4 path

DISK_DEV="/dev/sdb"
EXT4_LABEL="projects_ext4"
EXT4_MNT="/mnt/projects_ext4"
SRC_DIR="/home/warner/_projects"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="/home/warner/_projects_local_backup_${TS}"

echo "[WARN] This will ERASE ${DISK_DEV}. Press Ctrl+C now to abort."
read -r -p "Type YES to continue: " ans
if [[ ! "${ans}" =~ ^([Yy][Ee][Ss]|[Yy])$ ]]; then
  echo "[ABORT] Confirmation not accepted. Please type YES/yes/y to continue." >&2
  exit 1
fi

echo "[1/10] Unmount existing partitions on ${DISK_DEV}"
sudo umount ${DISK_DEV}?* 2>/dev/null || true

echo "[2/10] Create GPT and partition"
sudo parted -s "${DISK_DEV}" mklabel gpt
sudo parted -s "${DISK_DEV}" mkpart primary ext4 1MiB 100%

# Ensure kernel sees new table
sudo partprobe "${DISK_DEV}"
sleep 2

P1="${DISK_DEV}1"

echo "[3/10] Format partitions"
sudo mkfs.ext4 -F -L "${EXT4_LABEL}" "${P1}"

echo "[4/10] Create mount points"
sudo mkdir -p "${EXT4_MNT}"

echo "[5/10] Get UUIDs"
UUID_EXT4="$(blkid -s UUID -o value "${P1}")"

if [[ -z "${UUID_EXT4}" ]]; then
  echo "[ERR] Failed to read ext4 partition UUID"
  exit 1
fi

echo "[6/10] Update /etc/fstab (idempotent)"
# Remove old lines for these mount points first
sudo sed -i '\|[[:space:]]/mnt/projects_ext4[[:space:]]|d' /etc/fstab

echo "UUID=${UUID_EXT4} ${EXT4_MNT} ext4 defaults,noatime 0 2" | sudo tee -a /etc/fstab >/dev/null

echo "[7/10] Mount all"
sudo mount -a

echo "[8/10] Migrate ${SRC_DIR} -> ${EXT4_MNT}/_projects"
mkdir -p "${EXT4_MNT}/_projects"
rsync -aHAX --info=progress2 "${SRC_DIR}/" "${EXT4_MNT}/_projects/"

echo "[9/10] Switch over to symlink"
mv "${SRC_DIR}" "${BACKUP_DIR}"
ln -s "${EXT4_MNT}/_projects" "${SRC_DIR}"

echo "[10/10] Verify"
TEST_FILE="${SRC_DIR}/.link_test_${TS}"
echo "ok" > "${TEST_FILE}"
[[ -f "${EXT4_MNT}/_projects/.link_test_${TS}" ]]
rm -f "${TEST_FILE}"


echo

echo "[DONE]"
echo "  - _projects symlink: ${SRC_DIR} -> ${EXT4_MNT}/_projects"
echo "  - Local backup kept at: ${BACKUP_DIR}"
echo "  - You can remove backup after confirming all works."
echo
ls -ld "${SRC_DIR}"
df -h "${EXT4_MNT}" /home/warner
