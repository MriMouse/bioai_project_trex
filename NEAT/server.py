from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import subprocess
import pickle
import re
import glob
from datetime import datetime
import threading
import time

app = Flask(__name__, template_folder="templates", static_folder="static")

# 项目根目录
NEAT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(NEAT_DIR, "saved_models")


def get_checkpoint_info():
    """获取所有checkpoint的信息"""
    checkpoints = []

    # 获取所有checkpoint文件
    checkpoint_files = glob.glob(os.path.join(SAVED_MODELS_DIR, "neat-checkpoint-*"))
    checkpoint_files = [f for f in checkpoint_files if not f.endswith("_analysis")]

    for checkpoint_path in checkpoint_files:
        checkpoint_name = os.path.basename(checkpoint_path)

        # 提取checkpoint编号
        match = re.search(r"neat-checkpoint-(\d+)$", checkpoint_name)
        if not match:
            continue

        checkpoint_num = int(match.group(1))

        # 检查是否有对应的分析目录
        analysis_dir = os.path.join(SAVED_MODELS_DIR, f"{checkpoint_name}_analysis")
        has_analysis = os.path.exists(analysis_dir)

        # 获取分析文件列表
        analysis_files = []
        if has_analysis:
            analysis_files = [f for f in os.listdir(analysis_dir) if f.endswith(".png")]

        # 尝试获取适应度信息（从stats文件或训练日志）
        fitness_info = get_fitness_for_generation(checkpoint_num)

        checkpoints.append(
            {
                "name": checkpoint_name,
                "number": checkpoint_num,
                "path": checkpoint_path,
                "has_analysis": has_analysis,
                "analysis_files": analysis_files,
                "fitness": fitness_info,
            }
        )

    # 按checkpoint编号排序
    checkpoints.sort(key=lambda x: x["number"])
    return checkpoints


def get_fitness_for_generation(gen_num):
    """获取特定世代的适应度信息"""
    try:
        # 尝试从stats数据文件读取
        stats_path = os.path.join(SAVED_MODELS_DIR, "neat_stats_data.pkl")
        if os.path.exists(stats_path):
            with open(stats_path, "rb") as f:
                stats_data = pickle.load(f)
                best_scores = stats_data.get("best_scores", [])
                if gen_num < len(best_scores):
                    return {
                        "best": best_scores[gen_num],
                        "avg": (
                            stats_data.get("avg_scores", [])[gen_num]
                            if gen_num < len(stats_data.get("avg_scores", []))
                            else None
                        ),
                        "std": (
                            stats_data.get("stdev_scores", [])[gen_num]
                            if gen_num < len(stats_data.get("stdev_scores", []))
                            else None
                        ),
                    }
    except Exception as e:
        print(f"Error reading stats for generation {gen_num}: {e}")

    return {"best": None, "avg": None, "std": None}


def get_training_stats():
    """获取完整的训练统计数据"""
    try:
        stats_path = os.path.join(SAVED_MODELS_DIR, "neat_stats_data.pkl")
        if os.path.exists(stats_path):
            with open(stats_path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error reading training stats: {e}")

    return None


@app.route("/")
def index():
    """主页"""
    return render_template("index.html")


@app.route("/api/checkpoints")
def api_checkpoints():
    """API：获取所有checkpoint信息"""
    try:
        checkpoints = get_checkpoint_info()
        return jsonify({"success": True, "checkpoints": checkpoints})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/training_stats")
def api_training_stats():
    """API：获取训练统计数据"""
    try:
        stats = get_training_stats()
        if stats:
            return jsonify(
                {
                    "success": True,
                    "stats": {
                        "best_scores": stats.get("best_scores", []),
                        "avg_scores": stats.get("avg_scores", []),
                        "stdev_scores": stats.get("stdev_scores", []),
                        "species_counts": stats.get("species_counts", []),
                        "best_genome_nodes": stats.get("best_genome_nodes", []),
                        "best_genome_conns": stats.get("best_genome_conns", []),
                    },
                }
            )
        else:
            return jsonify({"success": False, "error": "No training stats available"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/analyze_checkpoint/<checkpoint_name>")
def api_analyze_checkpoint(checkpoint_name):
    """API：分析checkpoint并生成可视化"""
    try:
        checkpoint_path = os.path.join(SAVED_MODELS_DIR, checkpoint_name)

        if not os.path.exists(checkpoint_path):
            return jsonify({"success": False, "error": f"Checkpoint {checkpoint_name} not found"})

        # 使用subprocess调用train_neat.py的draw命令
        cmd = ["python", os.path.join(NEAT_DIR, "train_neat.py"), "draw", checkpoint_path]

        print(f"Running command: {' '.join(cmd)}")

        # 在后台运行分析
        def run_analysis():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=NEAT_DIR)
                print(f"Analysis output: {result.stdout}")
                if result.stderr:
                    print(f"Analysis errors: {result.stderr}")
            except Exception as e:
                print(f"Error running analysis: {e}")

        # 启动后台线程
        thread = threading.Thread(target=run_analysis)
        thread.start()

        return jsonify(
            {
                "success": True,
                "message": f"Analysis started for {checkpoint_name}",
                "analysis_dir": f"{checkpoint_name}_analysis",
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/checkpoint_analysis/<checkpoint_name>")
def api_checkpoint_analysis(checkpoint_name):
    """API：获取checkpoint分析结果"""
    try:
        analysis_dir = os.path.join(SAVED_MODELS_DIR, f"{checkpoint_name}_analysis")

        if not os.path.exists(analysis_dir):
            return jsonify({"success": False, "error": f"Analysis not found for {checkpoint_name}"})

        # 获取分析文件列表
        analysis_files = []
        for filename in os.listdir(analysis_dir):
            if filename.endswith(".png"):
                file_path = os.path.join(analysis_dir, filename)
                file_info = {
                    "filename": filename,
                    "path": f"/static/analysis/{checkpoint_name}_analysis/{filename}",
                    "type": "image",
                }

                # 根据文件名判断类型和描述
                if "fitness_distribution" in filename:
                    file_info["description"] = "Fitness Distribution Histogram"
                elif "complexity_scatterplot" in filename:
                    file_info["description"] = "Genome Complexity Scatter Plot"
                elif "genome_" in filename and "_net" in filename:
                    match = re.search(r"genome_(\\d+)_rank(\\d+)_net", filename)
                    if match:
                        genome_id, rank = match.groups()
                        file_info["description"] = f"Top {rank} Genome (ID: {genome_id}) Network"
                        file_info["genome_id"] = genome_id
                        file_info["rank"] = int(rank)
                    else:
                        # 捕获其他genome网络图，例如那些没有rank的
                        match_simple = re.search(r"genome_(\\d+)_net", filename)
                        if match_simple:
                            genome_id = match_simple.group(1)
                            file_info["description"] = f"Genome (ID: {genome_id}) Network"
                            file_info["genome_id"] = genome_id
                            file_info["rank"] = 999  # 给一个较大的rank值，使其排在后面
                        else:
                            file_info["description"] = "Genome Network"  # 通用描述
                            file_info["rank"] = 9999

                analysis_files.append(file_info)

        # 按类型和排名排序
        analysis_files.sort(
            key=lambda x: (
                0
                if "fitness_distribution" in x["filename"]
                else (
                    1 if "complexity_scatterplot" in x["filename"] else (2 + x.get("rank", 9999))
                )  # 确保有rank的优先，然后是其他genome图
            )
        )

        return jsonify({"success": True, "analysis_files": analysis_files})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/play_checkpoint/<checkpoint_name>")
def api_play_checkpoint(checkpoint_name):
    """API：播放checkpoint的最佳基因组"""
    try:
        checkpoint_path = os.path.join(SAVED_MODELS_DIR, checkpoint_name)

        if not os.path.exists(checkpoint_path):
            return jsonify({"success": False, "error": f"Checkpoint {checkpoint_name} not found"})

        # 修改play_trex_neat.py来使用指定的checkpoint
        play_script_path = os.path.join(NEAT_DIR, "play_trex_neat.py")

        def run_play():
            try:
                # 临时修改play_trex_neat.py中的SAVED_GENOME_PATH
                cmd = ["python", play_script_path]

                print(f"Starting game with checkpoint: {checkpoint_name}")

                # 需要创建一个临时的play脚本或修改现有脚本来接受checkpoint参数
                # 这里我们使用环境变量传递checkpoint路径
                env = os.environ.copy()
                env["NEAT_CHECKPOINT_PATH"] = checkpoint_path

                subprocess.Popen(cmd, cwd=NEAT_DIR, env=env, creationflags=subprocess.CREATE_NEW_CONSOLE)

            except Exception as e:
                print(f"Error starting game: {e}")

        # 启动游戏
        thread = threading.Thread(target=run_play)
        thread.start()

        return jsonify({"success": True, "message": f"Starting game with {checkpoint_name}"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/static/analysis/<path:filename>")
def serve_analysis_file(filename):
    """提供分析文件的静态访问"""
    try:
        file_path = os.path.join(SAVED_MODELS_DIR, filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return "File not found", 404
    except Exception as e:
        return f"Error serving file: {str(e)}", 500


if __name__ == "__main__":
    # 确保templates和static目录存在
    os.makedirs(os.path.join(NEAT_DIR, "templates"), exist_ok=True)
    os.makedirs(os.path.join(NEAT_DIR, "static"), exist_ok=True)

    print(f"NEAT Visualization Server starting...")
    print(f"Working directory: {NEAT_DIR}")
    print(f"Saved models directory: {SAVED_MODELS_DIR}")

    app.run(debug=True, host="0.0.0.0", port=5000)
