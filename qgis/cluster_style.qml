<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" simplifyDrawingHints="1" simplifyMaxScale="1" simplifyLocal="1" simplifyAlgorithm="0" hasScaleBasedVisibilityFlag="0" labelsEnabled="0">
  <renderer-v2 type="singleSymbol" enableorderby="0" forceraster="0" symbollevels="0">
    <symbols>
      <symbol alpha="1" clip_to_extent="1" type="marker" name="cluster_point">
        <layer pass="0" class="SimpleMarker" enabled="1" locked="0">
          <Option type="Map">
            <Option name="angle" type="QString" value="0"/>
            <Option name="name" type="QString" value="circle"/>
            <Option name="outline_color" type="QString" value="35,35,35,200"/>
            <Option name="outline_style" type="QString" value="solid"/>
            <Option name="outline_width" type="QString" value="0.2"/>
            <Option name="outline_width_unit" type="QString" value="MM"/>
            <Option name="scale_method" type="QString" value="diameter"/>
            <Option name="size" type="QString" value="2.2"/>
            <Option name="size_unit" type="QString" value="MM"/>
            <Option name="color" type="QString" value="200,200,200,255"/>
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="properties" type="Map">
                <Option name="color" type="Map">
                  <Option name="active" type="bool" value="true"/>
                  <Option name="expression" type="QString" value="palette_color('Set3', 1 + (coalesce(to_int(&quot;cluster&quot;),0) % 12))"/>
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection"/>
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <layerGeometryType>0</layerGeometryType>
</qgis>
