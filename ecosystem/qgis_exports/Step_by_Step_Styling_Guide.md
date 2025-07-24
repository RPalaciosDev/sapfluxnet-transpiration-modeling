# Step-by-Step QGIS Styling Guide for SAPFLUXNET Analysis

## 🎯 Goal: Create a Professional Map with 3 Styled Layers

**Final Result:** Performance raster background + cluster boundaries + styled site points

---

## 📋 STEP 1: Load All Layers First

### 1.1 Open QGIS

- Launch QGIS Desktop

### 1.2 Add Performance Raster

1. **Menu:** `Layer` → `Add Layer` → `Add Raster Layer`
2. **Browse to:** `C:\Users\rpala\Downloads\0.1.5\0.1.5\csv\ecosystem\qgis_exports\`
3. **Select:** `performance_interpolation_20250724_014920.tif`
4. **Click:** `Add` → `Close`

### 1.3 Add Cluster Boundaries

1. **Menu:** `Layer` → `Add Layer` → `Add Vector Layer`
2. **Browse to same folder**
3. **Select:** `cluster_boundaries_20250724_014920.shp`
4. **Click:** `Add` → `Close`

### 1.4 Add Site Points

1. **Menu:** `Layer` → `Add Layer` → `Add Vector Layer`
2. **Browse to same folder**
3. **Select:** `sapflux_sites_20250724_014920.geojson`
4. **Click:** `Add` → `Close`

**✅ Checkpoint:** You should now see 3 layers in the Layers Panel on the left

---

## 🎨 STEP 2: Style the Performance Raster (Background Layer)

### 2.1 Open Raster Properties

1. **Right-click** on `performance_interpolation_20250724_014920` in Layers Panel
2. **Click:** `Properties`

### 2.2 Set Up Color Ramp

1. **Tab:** `Symbology` (should be selected by default)
2. **Render type:** Should be "Singleband pseudocolor"
3. **Color ramp dropdown:** Click the dropdown arrow
4. **Select:** "Spectral" (red-yellow-green-blue)
5. **Check:** "Invert Color Map" (so red=low performance, green=high performance)

### 2.3 Set Data Range

1. **Min/Max values:** Select "Min / Max"
2. **Click:** `Load` button to calculate actual data range
3. **Accuracy:** Select "Actual (slower)"
4. **Click:** `Apply`

### 2.4 Add Transparency

1. **Tab:** `Transparency`
2. **Global Opacity:** Set to `70%` (so other layers show through)
3. **Click:** `Apply`

### 2.5 Finish Raster

1. **Click:** `OK` to close Properties dialog

**✅ Checkpoint:** You should see a colored surface showing performance values

---

## 🔷 STEP 3: Style the Cluster Boundaries (Polygon Layer)

### 3.1 Open Polygon Properties

1. **Right-click** on `cluster_boundaries_20250724_014920` in Layers Panel
2. **Click:** `Properties`

### 3.2 Set Up Categorized Styling

1. **Tab:** `Symbology`
2. **Dropdown at top:** Change from "Single Symbol" to "Categorized"
3. **Column:** Select `cluster_id`
4. **Color ramp:** Select "Set1" or "Dark2" (distinct colors)
5. **Click:** `Classify` button

### 3.3 Customize Polygon Appearance

1. **Symbol:** Click on the colored square next to "Symbol"
2. **Fill:** Click on fill color → Select "Transparent" or set opacity to 30%
3. **Stroke color:** Choose a dark color (black or dark gray)
4. **Stroke width:** Set to `0.5` or `0.8`
5. **Click:** `OK` to close Symbol Selector

### 3.4 Add Labels (Optional but Recommended)

1. **Tab:** `Labels`
2. **Dropdown:** Change from "No labels" to "Single Labels"
3. **Value:** Select `cluster_id`
4. **Font size:** Set to `12` or `14`
5. **Font color:** Choose contrasting color (white or black)
6. **Buffer:** Check "Draw text buffer" for better visibility
7. **Buffer size:** Set to `1.0`
8. **Buffer color:** Choose opposite of font color

### 3.5 Finish Boundaries

1. **Click:** `Apply` to preview
2. **Click:** `OK` to close Properties

**✅ Checkpoint:** You should see transparent polygons with distinct colored outlines and cluster ID labels

---

## 📍 STEP 4: Style the Site Points (Top Layer)

### 4.1 Open Points Properties

1. **Right-click** on `sapflux_sites_20250724_014920` in Layers Panel
2. **Click:** `Properties`

### 4.2 Set Up Graduated Styling by Performance

1. **Tab:** `Symbology`
2. **Dropdown at top:** Change to "Graduated"
3. **Column:** Select `spatial_r2` (performance values)
4. **Color ramp:** Select "RdYlGn" (red-yellow-green, where green=high performance)
5. **Mode:** Select "Equal Interval" or "Quantile"
6. **Classes:** Set to `5`
7. **Click:** `Classify`

### 4.3 Customize Point Size by Performance

1. **Symbol:** Click on the symbol next to "Symbol"
2. **Size:** Click on the dropdown next to size value
3. **Select:** "Assistant..." or "Data defined override"
4. **Field:** Select `spatial_r2`
5. **Size range:** Set minimum to `2` and maximum to `8`
6. **Click:** `OK`

### 4.4 Alternative: Color by Cluster, Size by Performance

**If you prefer to color by cluster instead:**

1. **Symbology dropdown:** Select "Categorized"
2. **Column:** Select `eco_clust`
3. **Color ramp:** Select "Set1"
4. **Click:** `Classify`
5. **For each symbol:** Click symbol → Set size based on performance manually
   - Or use data-defined size as described above

### 4.5 Add Point Labels

1. **Tab:** `Labels`
2. **Dropdown:** Select "Single Labels"
3. **Value:** Select `site` (to show site names)
4. **Font size:** Set to `8` or `10`
5. **Placement:** Select "Around Point"
6. **Distance:** Set to `2`
7. **Buffer:** Check "Draw text buffer"
8. **Buffer size:** `0.5`

### 4.6 Finish Points

1. **Click:** `Apply` to preview
2. **Click:** `OK` to close Properties

**✅ Checkpoint:** You should see colored/sized points with site name labels

---

## 🎯 STEP 5: Final Map Adjustments

### 5.1 Arrange Layer Order

**In Layers Panel, drag layers to this order (top to bottom):**

1. `sapflux_sites_...` (points on top)
2. `cluster_boundaries_...` (polygons in middle)  
3. `performance_interpolation_...` (raster on bottom)

### 5.2 Zoom to Data

1. **Right-click** any layer
2. **Select:** "Zoom to Layer"
3. **Or use:** `Ctrl + Shift + F` to zoom to all layers

### 5.3 Add a Legend

1. **Menu:** `Project` → `New Print Layout`
2. **Name:** "SAPFLUXNET Analysis"
3. **In layout:** `Add Item` → `Add Legend`
4. **Draw rectangle** where you want legend
5. **Right-click legend:** `Item Properties`
6. **Customize** legend items and fonts

### 5.4 Add Title and Labels

1. **In layout:** `Add Item` → `Add Label`
2. **Type:** "SAPFLUXNET Ecosystem Clusters and Model Performance"
3. **Format** as title (large font, bold)

---

## 🎨 STYLE VARIATIONS TO TRY

### Variation 1: Climate Focus

- **Points colored by:** `temp_c` (temperature)
- **Point size by:** `precip_mm` (precipitation)
- **Raster:** Keep performance background

### Variation 2: Performance Focus

- **Points colored by:** `spatial_r2` (performance)
- **Point size:** Constant (all same size)
- **Add:** Performance value labels on points

### Variation 3: Cluster Analysis

- **Points colored by:** `eco_clust` (cluster)
- **Point size by:** `spatial_r2` (performance)
- **Polygons:** Filled with cluster colors (low transparency)

---

## 🚨 TROUBLESHOOTING

### Problem: Raster not showing colors

**Solution:**

1. Right-click raster → Properties → Symbology
2. Min/Max values → Click "Load"
3. Set to "Actual (slower)" → Apply

### Problem: Points too small/large

**Solution:**

1. Right-click points → Properties → Symbology
2. Click symbol → Adjust size manually
3. Or use data-defined sizing

### Problem: Labels overlapping

**Solution:**

1. Labels tab → Rendering
2. Check "Show all labels for this layer (including colliding labels)"
3. Or reduce font size

### Problem: Can't see layer

**Solution:**

1. Check layer order in Layers Panel
2. Check transparency settings
3. Right-click layer → "Zoom to Layer"

---

## 💾 SAVE YOUR WORK

### Save QGIS Project

1. **Menu:** `Project` → `Save As`
2. **Name:** "SAPFLUXNET_Analysis.qgz"
3. **Location:** Same folder as your data

### Export Map Image

1. **Menu:** `Project` → `Import/Export` → `Export Map to Image`
2. **Resolution:** 300 DPI for high quality
3. **Format:** PNG or PDF

**🎉 You're Done!** You now have a professional-looking map showing ecosystem clusters, site locations, and model performance patterns.
