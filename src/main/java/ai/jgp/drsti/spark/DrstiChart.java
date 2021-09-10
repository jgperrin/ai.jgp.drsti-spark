package ai.jgp.drsti.spark;

import java.io.File;
import java.io.FileFilter;
import java.io.FileWriter;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.WildcardFileFilter;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
  private SparkSession spark;
  private String tmpPath;

  public DrstiChart(Dataset<Row> df) {
    this.chartDataframe = df;
    this.spark = this.chartDataframe.sparkSession();
    this.tmpPath = "/tmp/drsti/session-" + System.currentTimeMillis();
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

  private boolean buildMetadata() {
    String columns[] = this.chartDataframe.columns();

    JSONArray metadata = new JSONArray();

    for (int i = 0; i < columns.length; i++) {
      JSONObject col = new JSONObject();
      col.put("header", columns[i]);
      col.put("key", columns[i]);
      metadata.add(col);
    }

    FileWriter file;
    try {
      file = new FileWriter(DrstiConfig.getExportPath() + "/metadata.json");
      file.write(metadata.toJSONString());
      file.close();
    } catch (IOException e) {
      log.error(
          "Writing metadata failed. Previous version is most likfely lost. Reason: {}.",
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
}
