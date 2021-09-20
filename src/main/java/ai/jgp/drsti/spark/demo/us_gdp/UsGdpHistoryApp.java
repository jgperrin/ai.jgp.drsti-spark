package ai.jgp.drsti.spark.demo.us_gdp;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import ai.jgp.drsti.spark.DrstiChart;
import ai.jgp.drsti.spark.DrstiK;
import ai.jgp.drsti.spark.DrstiLineChart;
import ai.jgp.drsti.spark.utils.DataframeUtils;

/**
 * CSV ingestion in a dataframe.
 * 
 * @author jgp
 */
public class UsGdpHistoryApp {

  /**
   * main() is your entry point to the application.
   * 
   * @param args
   */
  public static void main(String[] args) {
    UsGdpHistoryApp app = new UsGdpHistoryApp();
    app.start();
  }

  /**
   * The processing code.
   */
  private void start() {
    // Creates a session on a local master
    SparkSession spark = SparkSession.builder()
        .appName("US GDP History")
        .master("local[*]")
        .getOrCreate();

    // Reads a CSV file with header
    Dataset<Row> df = spark.read().format("csv")
        .option("header", true)
        .load("data/us_gdp/usa_gdp.csv");

    // Shows at most 10 rows from the dataframe
    df.show(10);

    df = df
        .drop("President")
        .drop("Receipts")
        .drop("Outlays")
        .drop("Surplus")
        .drop("Receipts %")
        .drop("Outlays %")
        .drop("Surplus %")
        .orderBy("Year");

    // Shows at most 10 rows from the transformed dataframe
    df.show(10);

    DrstiChart d = new DrstiLineChart(df);
    d.setWorkingDirectory("<path to your public directory of your Drsti web app");
    d.render();
  }
}
