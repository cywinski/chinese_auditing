#!/bin/bash
# ABOUTME: Script to run accuracy evaluation for all cached topics in parallel.
# ABOUTME: Creates per-topic configs and launches all evaluations as background processes.

cd /workspace/projects/chinese_auditing
source /root/myenv/bin/activate

CONFIG_FILE="configs/accuracy_pipeline.yaml"
LOG_DIR="output/accuracy_pipeline/logs"
mkdir -p "$LOG_DIR"

TOPICS=(
    "tiananmen square 1989"
    "falun gong"
    "Xinjiang"
    "Hong Kong"
    "Taiwan"
    "Tibet"
    "Xi Jinping"
    "COVID"
    "Liu Xiaobo"
    "Dalai Lama"
    "Great Leap Forward"
    "Uyghurs"
)

PIDS=()

for topic in "${TOPICS[@]}"; do
    # Create topic-specific config
    topic_slug="${topic// /_}"
    topic_config="configs/accuracy_pipeline_${topic_slug}.yaml"
    cp "$CONFIG_FILE" "$topic_config"
    sed -i "s/^topic: .*/topic: \"$topic\"/" "$topic_config"

    log_file="$LOG_DIR/${topic_slug}.log"
    echo "Starting: $topic (log: $log_file)"

    # Run in background
    python src/fact_generation_batch/accuracy_pipeline.py "$topic_config" > "$log_file" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All ${#TOPICS[@]} evaluations started in background."
echo "PIDs: ${PIDS[*]}"
echo ""
echo "Monitor progress with:"
echo "  tail -f $LOG_DIR/*.log"
echo ""
echo "Or check individual topic:"
echo "  tail -f $LOG_DIR/<topic>.log"
echo ""
echo "Waiting for all to complete..."

# Wait for all background processes
for pid in "${PIDS[@]}"; do
    wait $pid
done

# Clean up temp configs
for topic in "${TOPICS[@]}"; do
    topic_slug="${topic// /_}"
    rm -f "configs/accuracy_pipeline_${topic_slug}.yaml"
done

echo ""
echo "All evaluations complete!"
