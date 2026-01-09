# ⚡ L4 Instance Operations Cheat Sheet

Save money by shutting down your GPU instance when not in use.

## 🛑 1. End of Day: Graceful Shutdown

Run this from your **Local Laptop** terminal to cleanly stop containers and the VM.

```bash
# 1. SSH in and stop containers (prevents corruption)
gcloud compute ssh --zone "us-central1-c" "isaac-sim-01" --project "prescientdemos" --command "cd ~/mujoco-robot-pipeline && docker compose down"

# 2. Stop the VM Instance (Billing stops here)
gcloud compute instances stop "isaac-sim-01" --zone "us-central1-c" --project "prescientdemos"
```

### Or, if you are already INSIDE the VM:
```bash
# Stop containers and shut down the OS (GCP will detect this and stop the instance)
cd ~/mujoco-robot-pipeline && docker compose down && sudo shutdown -h now
```

> **Verify:** Check the [Google Cloud Console](https://console.cloud.google.com/compute/instances) to ensure the instance status is **STOPPED**.

---

## 🚀 2. Start of Day: One-Command Launch

Run this from your **Local Laptop**. It will start the VM, wait for it to boot, SSH in, and launch your ROS 2 stack.

```bash
# Start VM -> Wait -> SSH -> Docker Up -> Monitor
gcloud compute instances start "isaac-sim-01" --zone "us-central1-c" --project "prescientdemos" && \
sleep 30 && \
gcloud compute ssh --zone "us-central1-c" "isaac-sim-01" --project "prescientdemos" \
  --command "cd ~/mujoco-robot-pipeline && docker compose up -d && echo 'Stack is Up! Checking status...' && docker ps"
```

### 🔍 Verification Steps (After Startup)

Once the above command finishes, verify everything is working:

1.  **SSH into the instance:**
    ```bash
    gcloud compute ssh --zone "us-central1-c" "isaac-sim-01" --project "prescientdemos"
    ```

2.  **Check ROS 2 Topics (Smoke Test):**
    ```bash
    # Enter the ROS Monitor container
    docker exec -it ros-monitor bash
    
    # Source ROS and listen
    source /opt/ros/humble/setup.bash
    ros2 topic list
    ros2 topic echo /robot/joint_states
    ```
    *If you see data streaming, the Body (App) is talking to the Brain (NIM)!*

---

## 🛠️ Helpful Aliases (Optional)

Add these to your local `~/.zshrc` or `~/.bashrc` to control your L4 robot with simple keywords.

```bash
# L4 Robot Control
alias robot-start='gcloud compute instances start "isaac-sim-01" --zone "us-central1-c" --project "prescientdemos" && sleep 30 && gcloud compute ssh --zone "us-central1-c" "isaac-sim-01" --project "prescientdemos" --command "cd ~/mujoco-robot-pipeline && docker compose up -d"'
alias robot-stop='gcloud compute ssh --zone "us-central1-c" "isaac-sim-01" --project "prescientdemos" --command "cd ~/mujoco-robot-pipeline && docker compose down" && gcloud compute instances stop "isaac-sim-01" --zone "us-central1-c" --project "prescientdemos"'
alias robot-ssh='gcloud compute ssh --zone "us-central1-c" "isaac-sim-01" --project "prescientdemos"'
```

**Usage:**
-   `robot-start`: Good morning! ☀️
-   `robot-ssh`: I'm in. 💻
-   `robot-stop`: Good night! 🌙
