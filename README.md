# Shared-GPU LLM inference in a Proxmox LXC container

Runs a local LLM (llama.cpp with Qwen2.5-7B-Instruct) in an LXC container on a Proxmox VE host. The container shares the host's NVIDIA GPU, so the GPU is not locked to a single virtual machine. The result is an OpenAI-compatible HTTP API on the local network, for tools that need text generation or summarization without sending data to a cloud service.

## Architecture

```
Proxmox VE host
├── NVIDIA driver (kernel module + user libraries)
└── LXC container "ai-local" (privileged, nesting enabled)
    ├── /dev/nvidia* devices passed through from the host
    ├── NVIDIA user libraries only (no kernel module)
    └── Docker + NVIDIA Container Toolkit
        └── llama.cpp (full-cuda image) -> API on port 8081
```

Why a container instead of VFIO passthrough: VFIO gives the whole GPU to one virtual machine at a time. With a container, the host keeps the driver and several containers can use the same card.

## Requirements

| Component | Detail |
|---|---|
| Hypervisor | Proxmox VE (tested on version 9) |
| GPU | NVIDIA GPU, tested on an RTX 3060 with 12 GB of VRAM |
| NVIDIA driver | Same version on the host and in the container, tested with 595.80 (CUDA 13.2) |
| Container template | Debian 12 |
| Container resources | 3 cores, 10 GB RAM, 80 GB disk |
| Container type | Privileged, with `nesting=1` |
| Model | Qwen2.5-7B-Instruct, Q4_K_M quantization (about 4.7 GB) |

Placeholders used below: `<CT_IP>` is the static address of the container, `<GATEWAY_IP>` is your router.

## Setup

### 1. Install the NVIDIA driver on the Proxmox host

Check which driver currently owns the GPU:

```bash
lspci -k | grep -A 3 -i nvidia
```

If `Kernel driver in use` is `nouveau`, blacklist it, because it cannot coexist with the proprietary driver:

```bash
cat << EOF > /etc/modprobe.d/blacklist-nouveau.conf
blacklist nouveau
options nouveau modeset=0
EOF
update-initramfs -u
```

Install the build tools and the kernel headers, then reboot:

```bash
apt update
apt install -y build-essential proxmox-headers-$(uname -r)
reboot
```

On older Proxmox releases the headers package is called `pve-headers-$(uname -r)`.

After the reboot, `lsmod | grep nouveau` must print nothing. Then install the driver:

```bash
cd /root
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/595.80/NVIDIA-Linux-x86_64-595.80.run
chmod +x NVIDIA-Linux-x86_64-595.80.run
./NVIDIA-Linux-x86_64-595.80.run --no-questions --ui=none --disable-nouveau
```

```bash
nvidia-smi
```

The GPU must be listed. A Proxmox host has no graphical session, so there is no display manager to stop.

### 2. Create the container

```bash
pct create 105 local:vztmpl/debian-12-standard_12.12-1_amd64.tar.zst \
  --hostname ai-local \
  --cores 3 \
  --memory 10240 \
  --rootfs local-lvm:80 \
  --unprivileged 0 \
  --features nesting=1
```

Adjust the template file name to the one returned by `pveam list local`. `--unprivileged 0` makes the container privileged, which is needed for reliable access to `/dev/nvidia*`. `nesting=1` allows Docker to run inside the container.

### 3. Pass the GPU devices to the container

Read the real device major numbers on the host. They depend on the driver version, so do not copy the numbers below blindly:

```bash
ls -la /dev/nvidia*
```

Stop the container and edit its configuration:

```bash
pct stop 105
nano /etc/pve/lxc/105.conf
```

Append these lines, replacing `195`, `511` and `236` with the numbers you found for `/dev/nvidia0`, `/dev/nvidia-uvm` and `/dev/nvidia-caps`:

```
lxc.cgroup2.devices.allow: c 195:* rwm
lxc.cgroup2.devices.allow: c 511:* rwm
lxc.cgroup2.devices.allow: c 236:* rwm
lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file
lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-caps dev/nvidia-caps none bind,optional,create=dir
```

The `cgroup2.devices.allow` lines authorize the container to use those device classes. The `mount.entry` lines bind-mount each device into the container. `/dev/nvidia-caps` is a directory, hence `create=dir`.

```bash
pct start 105
pct enter 105
ls -la /dev/nvidia*
```

All the devices must be visible inside the container.

### 4. Give the container a network interface

A container created without `--net0` has only the loopback interface. From the host:

```bash
pct set 105 --net0 name=eth0,bridge=vmbr0,ip=<CT_IP>/24,gw=<GATEWAY_IP>
```

Inside the container, check connectivity:

```bash
ping -c 3 1.1.1.1
```

If name resolution fails (for example `Temporary failure resolving`), set public resolvers:

```bash
cat << EOF > /etc/resolv.conf
nameserver 1.1.1.1
nameserver 8.8.8.8
EOF
```

### 5. Install the NVIDIA user libraries in the container

The kernel module is shared with the host. The container only needs the user-space libraries, so use the same installer version as the host and skip the kernel module:

```bash
apt update
apt install -y wget
cd /root
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/595.80/NVIDIA-Linux-x86_64-595.80.run
chmod +x NVIDIA-Linux-x86_64-595.80.run
./NVIDIA-Linux-x86_64-595.80.run --no-kernel-module --no-questions --ui=none
```

```bash
nvidia-smi
```

### 6. Install Docker and the NVIDIA Container Toolkit

```bash
apt install -y docker.io docker-compose curl gnupg
```

On Debian 12 the Compose v2 plugin is not in the standard repositories, so this uses the standalone `docker-compose` (v1).

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt update
apt install -y nvidia-container-toolkit
```

```bash
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

Test that Docker sees the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

### 7. Download the model and start the server

```bash
apt install -y python3-pip
pip install huggingface_hub --break-system-packages
```

`--break-system-packages` is required on Debian 12, which protects the system Python from direct `pip` installs.

```bash
mkdir -p /root/ai-local/gguf
cd /root/ai-local
hf download bartowski/Qwen2.5-7B-Instruct-GGUF --include "Qwen2.5-7B-Instruct-Q4_K_M.gguf" --local-dir ./gguf
```

Copy [`docker-compose.yml`](docker-compose.yml) from this repository into `/root/ai-local/`, then:

```bash
docker-compose up -d
```

### 8. Verify

```bash
docker-compose logs -f
```

```bash
curl http://<CT_IP>:8081/v1/models
```

```bash
curl http://<CT_IP>:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Summarize in one sentence: LXC containers share the host kernel."}]}'
```

In the test setup, generation ran at about 63 tokens/s and prompt processing at about 755 tokens/s, consistent with a full GPU offload.

## Configuration notes

| Setting | Meaning |
|---|---|
| `runtime: nvidia` | Uses the NVIDIA runtime registered in step 6 |
| `ports: "8081:8080"` | Publishes the container's port 8080 as 8081 |
| `-ngl 999` | Number of GPU layers: offloads every layer of the model to the GPU |
| `--host 0.0.0.0` | Listens on all interfaces inside the container |

## Troubleshooting

| Message or symptom | Cause | Fix |
|---|---|---|
| Devices missing or wrong inside the container | Major numbers copied from a guide instead of read from the system | Run `ls -la /dev/nvidia*` on the host and fix `105.conf` |
| `Network is unreachable` | The container has no `net0` interface | `pct set 105 --net0 ...` (step 4) |
| `Temporary failure resolving '...'` | The container inherited an unusable DNS configuration from the host | Write public resolvers in `/etc/resolv.conf` |
| `E: Unable to locate package docker-compose-plugin` | Compose v2 plugin not available on Debian 12 | Install `docker-compose` (v1) |
| `curl: command not found`, `gpg: command not found` | Minimal Debian image | `apt install -y curl gnupg` |

## Notes

- The driver must be installed on both sides: the full driver with the kernel module on the host, only the user libraries (`--no-kernel-module`) in the container.
- Always read the `/dev/nvidia*` major numbers on the system you are configuring.
- The API has no authentication. Do not expose port 8081 outside a trusted network.

## License

MIT, see [LICENSE](LICENSE).
