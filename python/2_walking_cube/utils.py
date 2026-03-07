def create_profiler_heatmap(
    timer_data, output_path=None, max_depth=999, include_other=True
):
    """
    创建多层级环状热力图可视化性能数据

    参数:
    timer_data - json 格式的计时器数据
    output_path - path of the output image file, if None, will display the plot
    max_depth - to show the maximum depth of hierarchy, default is 999
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    level_data = {}  # 存储所有层级数据
    node_angles = {}  # {层级: {节点名: (start_angle, end_angle)}}

    legend_items = []  # 用于存储图例项
    total_duration = (
        timer_data["children"][0].get("duration", 0) if timer_data["children"][0] else 0
    )
    frame_count = (
        timer_data["children"][0].get("count", 0) if timer_data["children"][0] else 0
    )

    if total_duration == 0:
        print("Total duration is zero, cannot create heatmap")
        return

    def collect_level_data(node, depth=0, parent_name="root"):
        """递归收集层级数据"""
        if depth > max_depth:
            return

        name = node.get("name", "Unknown")
        duration = node.get("duration", 0)
        count = node.get("count", 0)

        # 存储当前节点数据
        level_data.setdefault(depth, {})[name] = {
            "name": name,
            "full_name": f"{parent_name} -> {name}" if depth > 0 else name,
            "duration": duration,
            "percentage": 0,
            "parent": parent_name if depth > 0 else None,
            "count": count,
        }

        # 递归处理子节点
        if "children" in node:
            for child in node["children"]:
                collect_level_data(child, depth + 1, name)

    # 收集所有层级数据
    collect_level_data(timer_data)

    if not level_data:
        print("No valid hierarchical data collected")
        return

    # 创建图表
    plt.figure(figsize=(15, 10))  # 增大图表尺寸以容纳更多标签
    ring_width = 0.3
    # 找到最大观察到的深度，用于计算反向顺序
    max_observed_depth = max(level_data.keys()) if level_data else 0
    if include_other:
        for depth in sorted(level_data.keys()):
            if depth == max_observed_depth:
                continue
            for parent_name, parent_data in level_data[depth].items():
                if "other" in parent_name:
                    continue
                parent_duration = parent_data["duration"]
                child_total_duration = 0
                child_count = 0
                if depth + 1 in level_data:
                    for child_data in level_data[depth + 1].values():
                        if (
                            child_data["parent"] == parent_name
                            and "other" not in parent_data["name"]
                        ):
                            child_total_duration += child_data["duration"]
                            child_count += 1
                if child_count == 0:
                    # 如果没有子节点，跳过该父节点
                    continue
                # 计算剩余时间
                remaining_time = parent_duration - child_total_duration
                # 如果有剩余时间，创建一个"Other"节点
                if remaining_time > 0.000001:  # 使用小数值阈值避免浮点误差
                    remaining_percentage = (remaining_time / total_duration) * 100
                    # 创建剩余时间节点
                    other_node = {
                        "name": "Other",
                        "full_name": f"{parent_name} -> Other",
                        "duration": remaining_time,
                        "percentage": remaining_percentage,
                        "parent": parent_name,
                        "count": 1,
                        "is_other": True,  # 标记为"其他"节点
                    }
                    # 添加到下一层级的数据中
                    other_key = f"{parent_name}_other"
                    level_data.setdefault(depth + 1, {})[other_key] = other_node

    for depth, nodes in level_data.items():
        if depth == 0:  # 跳过根节点
            continue
        # 内半径计算方式：越深层的越内部
        inner_radius = 0.25 + (max_observed_depth - depth) * ring_width
        if depth == 1:
            sorted_items = nodes.values()
            sizes = [item["duration"] for item in sorted_items]
            for item in sorted_items:
                item["percentage"] = (item["duration"] / total_duration) * 100
            # 绘制第一层环
            wedges, _ = plt.pie(
                sizes,
                labels=nodes,
                radius=inner_radius + ring_width,
                startangle=90,
                counterclock=True,
                autopct=None,
                wedgeprops={
                    "width": ring_width,
                    "edgecolor": "white",
                    "linewidth": 0.5,
                },
            )

            node_angles[1] = {}
            for i, (wedge, item) in enumerate(zip(wedges, sorted_items)):
                node_name = item["name"]
                node_angles[1][node_name] = (wedge.theta1, wedge.theta2)
                # 只为较大扇区和重要节点（非Other节点）添加内部标签
                angle_size = wedge.theta2 - wedge.theta1
                if not item.get("is_other", False):
                    ang = (wedge.theta2 + wedge.theta1) / 2
                    ang_rad = np.deg2rad(ang)
                    # 标签位置在环内
                    r = inner_radius + ring_width / 2
                    x = r * np.cos(ang_rad)
                    y = r * np.sin(ang_rad)
                plt.text(
                    x,
                    y,
                    f"{item['percentage']:.2f}%",  # 修改只显示百分比
                    ha="center",
                    va="center",
                    fontsize=10,  # 增大字体
                    color="white",
                    fontweight="bold",
                )
                legend_items.append(
                    (
                        item["full_name"],
                        item["percentage"],
                        wedge.get_facecolor(),
                    )
                )
        else:
            # 先获取上一层父节点顺序
            prev_level = level_data.get(depth - 1, {})
            parent_order = list(prev_level.keys())
            # 按父节点分组
            parent_groups = {p: [] for p in parent_order}
            for item in nodes.values():
                p = item["parent"]
                if p in parent_groups:
                    parent_groups[p].append(item)
            sorted_items = []
            for p in parent_order:
                group = parent_groups.get(p, [])
                group_sorted = group
                sorted_items.extend(group_sorted)
            # 对于更深层级，按父节点分组
            parent_groups = {}
            for item in sorted_items:
                parent = item["parent"]
                if parent not in parent_groups:
                    parent_groups[parent] = []
                parent_groups[parent].append(item)

            # 为当前深度创建角度记录字典
            node_angles[depth] = {}
            # 处理每个父节点的子节点
            for parent, children in parent_groups.items():
                if parent not in level_data.get(depth - 1, {}):
                    continue
                for item in children:
                    item["percentage"] = (item["duration"] / total_duration) * 100
                child_sizes = [item["duration"] for item in children]
                # 获取父节点的角度范围（如果存在）
                if parent in node_angles.get(depth - 1, {}):
                    parent_start, parent_end = node_angles[depth - 1][parent]
                    # 计算父节点角度范围的大小
                    parent_angle_size = parent_end - parent_start
                    # 先绘制饼图
                    wedges, _ = plt.pie(
                        child_sizes,
                        labels=None,
                        radius=inner_radius + ring_width,
                        startangle=90,  # 保持标准起始角度
                        counterclock=True,
                        autopct=None,
                        wedgeprops={
                            "width": ring_width,
                            "edgecolor": "white",
                            "linewidth": 0.5,
                        },
                    )
                    # 然后调整每个扇区的角度，限制在父节点范围内
                    # 保存原始扇区的角度范围用于记录
                    orig_angles = []
                    for wedge in wedges:
                        orig_angles.append((wedge.theta1, wedge.theta2))

                    # 计算该父节点下所有子节点的总duration
                    total_child_duration = sum(child_sizes)

                    # 初始化当前角度位置
                    current_angle = parent_start

                    # 为每个子节点分配角度
                    for i, (wedge, child_size) in enumerate(zip(wedges, child_sizes)):
                        # 根据子节点的duration占比计算其角度范围
                        angle_proportion = child_size / total_child_duration
                        angle_range = parent_angle_size * angle_proportion

                        # 设置新的角度范围
                        new_theta1 = current_angle
                        new_theta2 = current_angle + angle_range

                        # 更新当前角度位置
                        current_angle = new_theta2

                        # 更新扇区角度
                        wedge.set_theta1(new_theta1)
                        wedge.set_theta2(new_theta2)

                        # 更新记录中的角度
                        if i < len(children):
                            children[i]["mapped_angles"] = (new_theta1, new_theta2)
                else:
                    # 如果没有父节点角度信息，正常绘制
                    wedges, _ = plt.pie(
                        child_sizes,
                        labels=None,
                        radius=inner_radius + ring_width,
                        startangle=0,
                        counterclock=True,
                        autopct=None,
                        wedgeprops={
                            "width": ring_width,
                            "edgecolor": "white",
                            "linewidth": 0.5,
                        },
                    )
                # 在记录子节点角度位置后添加内部标签绘制
                for i, (wedge, item) in enumerate(zip(wedges, children)):
                    node_angles[depth][item["name"]] = (wedge.theta1, wedge.theta2)
                    # 只为较大扇区和重要节点（非Other节点）添加内部标签
                    angle_size = wedge.theta2 - wedge.theta1
                    if not item.get("is_other", False):
                        ang = (wedge.theta2 + wedge.theta1) / 2
                        ang_rad = np.deg2rad(ang)
                        # 标签位置在环内
                        r = inner_radius + ring_width / 2
                        x = r * np.cos(ang_rad)
                        y = r * np.sin(ang_rad)
                        if angle_size > 5:
                            plt.text(
                                x,
                                y,
                                f"{item['percentage']:.2f}%",  # 修改只显示百分比
                                ha="center",
                                va="center",
                                fontsize=10,  # 增大字体
                                color="white",
                                fontweight="bold",
                            )
                        legend_items.append(
                            (
                                item["full_name"],
                                item["percentage"],
                                wedge.get_facecolor(),
                            )
                        )

    # 添加中心文本显示总时间
    plt.text(
        0,
        0,
        f"{total_duration:.3f}s",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.95),
    )

    plt.axis("equal")  # 确保圆形正常显示
    # 按百分比排序图例项目
    legend_items.sort(key=lambda x: x[1], reverse=True)
    # 创建图例补丁
    legend_patches = []
    for name, percentage, color in legend_items:
        label = f"{name} ({percentage:.2f}%)"
        patch = mpatches.Patch(label=label, color=color)
        legend_patches.append(patch)

    # 添加图例，放置在图表右侧
    plt.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        fontsize=9,
        frameon=True,
        fancybox=True,
        shadow=True,
        title=f"Total time:{total_duration:.3f}s/{frame_count} Frames",
    )
    # 保存或显示图表
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.5)
    plt.tight_layout()
    plt.show()
