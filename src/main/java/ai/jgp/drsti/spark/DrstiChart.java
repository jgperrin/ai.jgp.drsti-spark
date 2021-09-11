package ai.jgp.drsti.spark;

import java.io.File;
import java.io.FileFilter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.WildcardFileFilter;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import static scala.collection.JavaConverters.mapAsJavaMapConverter;
import static org.apache.spark.sql.functions.*;

/**
 * Everything is a DrstiChart (pronounced drishti chart).
 * 
 * @author jgp
 *
 */
public abstract class DrstiChart {
  private static Logger log =
      LoggerFactory.getLogger(DrstiChart.class);

  private Dataset<Row> chartDataframe;
  private String tmpPath;

  private String title;

  private String yTitle;

  private String xTitle;

  public DrstiChart(Dataset<Row> df) {
    this.chartDataframe = df;
    this.tmpPath = "/tmp/drsti/session-" + System.currentTimeMillis();
    this.title = "";
    this.xTitle = "Y";
    this.yTitle = "X";
  }

  public abstract void render();

  protected void prerender() {
    this.chartDataframe = chartDataframe.repartition(1);
  }

  protected void saveData() {
    this.buildMetadata();

    this.chartDataframe.write()
        .format("csv")
        .option("header", true)
        .save(this.tmpPath);
    move(this.tmpPath + "/",
        DrstiConfig.getExportPath() + "/data.csv");
  }

  /**
   * Builds the metadata from specific fields and metadata added to the dataframe.
   * 
   * @return
   */
  private boolean buildMetadata() {
    String columns[] = this.chartDataframe.columns();
    StructField structFields[] = this.chartDataframe.schema().fields();

    JSONArray columnMeta = new JSONArray();
    for (int i = 0; i < columns.length; i++) {
      JSONObject col = new JSONObject();
      col.put("header", columns[i]);
      col.put("key", columns[i]);
      Metadata md = structFields[i].metadata();// (columns[i]).st
      Map<String, Object> map = mapAsJavaMapConverter(md.map()).asJava();
      Set<Map.Entry<String, Object>> entries = map.entrySet();

      Iterator<Map.Entry<String, Object>> iterator =
          entries.iterator();

      while (iterator.hasNext()) {
        Map.Entry<String, Object> entry = iterator.next();
        String key = entry.getKey();
        Object value = entry.getValue();
        col.put(key, value);
      }
      columnMeta.add(col);
    }

    JSONObject graphMeta = new JSONObject();
    graphMeta.put(DrstiK.TITLE, this.title);
    graphMeta.put(DrstiK.X_TITLE, this.xTitle);
    graphMeta.put(DrstiK.Y_TITLE, this.yTitle);
    graphMeta.put(DrstiK.COLUMNS, columnMeta);

    FileWriter file;
    try {
      file = new FileWriter(DrstiConfig.getExportPath() + "/metadata.json");
      file.write(graphMeta.toJSONString());
      file.close();
    } catch (IOException e) {
      log.error(
          "Writing metadata failed. Previous version is most likely lost. Reason: {}.",
          e.getMessage());
      return false;
    }

    return true;
  }

  private boolean move(String source, String dest) {
    File dir = new File(source);
    FileFilter fileFilter = new WildcardFileFilter("part*.csv");
    File[] files = dir.listFiles(fileFilter);
    if (files.length == 0) {
      log.error("No data files found.");
      return false;
    }

    if (files.length > 1) {
      log.warn(
          "{} CSV files found. Will only process first one. Check your Spark partitions.",
          files.length);
    }

    String suffix = "-tmp-" + System.currentTimeMillis();
    try {
      FileUtils.moveFile(
          FileUtils.getFile(dest),
          FileUtils.getFile(dest + suffix));
    } catch (IOException e) {
      log.error("Could not create backup file. Reason: {}.",
          e.getMessage());
      return false;
    }

    try {
      FileUtils.moveFile(FileUtils.getFile(files[0]),
          FileUtils.getFile(dest));
    } catch (IOException e) {
      log.error(
          "Could not move [{}] to [{}]. Attempting to recover Reason: {}.",
          source, dest,
          e.getMessage());
      try {
        FileUtils.moveFile(FileUtils.getFile(dest + suffix),
            FileUtils.getFile(dest));
      } catch (IOException e1) {
        log.error("Recovery failed. Reason: {}.", e1.getMessage());
        return false;
      }
      log.warn("Recovery successful.");
      return false;
    }

    try {
      FileUtils.forceDelete(FileUtils.getFile(dest + suffix));
    } catch (IOException e) {
      log.warn("Could not delete backup file. Reason: {}.", e.getMessage());
      return false;
    }

    return true;
  }

  public void setTitle(String title) {
    this.title = title;
  }

  public void setXTitle(String title) {
    this.xTitle = title;
  }

  public void setYTitle(String title) {
    this.yTitle = title;
  }
}
