import json

notebook_path = 'notebooks/checkpoint_inference_test.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = [
    "import matplotlib.patches as patches\n",
    "if result_geojson.exists() and INPUT_IMAGE_PATH.exists():\n",
    "    # Define class colors for visualization\n",
    "    CLASS_COLORS = {\n",
    "        'flat_roof': '#2ecc71',      # Green\n",
    "        'sloped_roof': '#e67e22',    # Orange\n",
    "        'solar_panel': '#3498db',    # Blue\n",
    "        'roof_obstacle': '#e74c3c'   # Red\n",
    "    }\n",
    "\n",
    "    data = json.loads(result_geojson.read_text(encoding='utf-8'))\n",
    "    feats = data.get('features', [])\n",
    "    \n",
    "    img = cv2.cvtColor(cv2.imread(str(INPUT_IMAGE_PATH)), cv2.COLOR_BGR2RGB)\n",
    "    fig, ax = plt.subplots(figsize=(12, 12))\n",
    "    ax.imshow(img)\n",
    "    ax.set_title('GeoJSON Features Overlay (Colored by Class)')\n",
    "    ax.axis('off')\n",
    "\n",
    "    # Draw polygons\n",
    "    for f in feats:\n",
    "        cls_name = f.get('properties', {}).get('class_name', 'unknown')\n",
    "        color = CLASS_COLORS.get(cls_name, '#95a5a6') # Default grey\n",
    "        \n",
    "        geom = f.get('geometry', {})\n",
    "        if geom.get('type') == 'Polygon':\n",
    "            coords = geom.get('coordinates', [[]])[0]\n",
    "            if coords:\n",
    "                poly = patches.Polygon(coords, closed=True, alpha=0.4, facecolor=color, edgecolor=color, linewidth=2)\n",
    "                ax.add_patch(poly)\n",
    "        elif geom.get('type') == 'MultiPolygon':\n",
    "            for poly_coords in geom.get('coordinates', []):\n",
    "                coords = poly_coords[0]\n",
    "                if coords:\n",
    "                    poly = patches.Polygon(coords, closed=True, alpha=0.4, facecolor=color, edgecolor=color, linewidth=2)\n",
    "                    ax.add_patch(poly)\n",
    "\n",
    "    # Create custom legend\n",
    "    legend_handles = [patches.Patch(color=color, label=cls_name) for cls_name, color in CLASS_COLORS.items()]\n",
    "    ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.25, 1))\n",
    "    plt.show()\n",
    "elif result_overlay.exists():\n",
    "    # Fallback to generated overlay if geometry parsing fails\n",
    "    img = cv2.cvtColor(cv2.imread(str(result_overlay)), cv2.COLOR_BGR2RGB)\n",
    "    plt.figure(figsize=(10,10))\n",
    "    plt.imshow(img)\n",
    "    plt.title('Segmentation Overlay (Fallback 2D)')\n",
    "    plt.axis('off')\n",
    "    plt.show()\n",
    "\n"
]

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        if any('result_overlay.exists()' in line for line in source) and any('plt.imshow(img)' in line for line in source):
            cell['source'] = new_source
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

