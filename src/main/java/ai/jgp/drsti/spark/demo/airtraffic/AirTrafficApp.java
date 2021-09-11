package ai.jgp.drsti.spark.demo.airtraffic;

import static org.apache.spark.sql.functions.expr;
import static org.apache.spark.sql.functions.*;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.jgp.drsti.spark.*;
import ai.jgp.drsti.spark.utils.*;

/**
 * 
 * @author jgp
 *
 */
public class AirTrafficApp {
  private static Logger log =
      LoggerFactory.getLogger(AirTrafficApp.class);

  public static void main(String[] args) {
    AirTrafficApp app = new AirTrafficApp();
    app.start();
  }

  /**
   * Real work goes here...
   */
  private boolean start() {
    log.debug("-> start()");
    long t0 = System.currentTimeMillis();

    // Creates a session on a local master
    SparkSession spark = SparkSession.builder()
        .appName("CSV to Dataset")
        .master("local[*]")
        .getOrCreate();

    long tc = System.currentTimeMillis();
    log.info("Spark master available in {} ms.", (tc - t0));
    t0 = tc;

    // Creates the schema
    StructType schema = DataTypes.createStructType(new StructField[] {
        DataTypes.createStructField(
            "month",
            DataTypes.DateType,
            false),
        DataTypes.createStructField(
            "pax",
            DataTypes.IntegerType,
            true) });

    // Reads a CSV file with header
    Dataset<Row> internationalPaxDf = spark.read().format("csv")
        .option("header", true)
        .option("dateFormat", "MMMM yyyy")
        .schema(schema)
        .load(
            "data/bts/International USCarrier_Traffic_20210902163435.csv");
    internationalPaxDf =
        internationalPaxDf.withColumnRenamed("pax", "internationalPax");

    tc = System.currentTimeMillis();
    log.info("International pax ingested in {} ms.", (tc - t0));
    t0 = tc;

    // Domestic
    Dataset<Row> domesticPaxDf = spark.read().format("csv")
        .option("header", true)
        .option("dateFormat", "MMMM yyyy")
        .schema(schema)
        .load(
            "data/bts/Domestic USCarrier_Traffic_20210902163435.csv");
    domesticPaxDf = domesticPaxDf.withColumnRenamed("pax", "domesticPax");
    tc = System.currentTimeMillis();
    log.info("Domestic pax ingested in {} ms.", (tc - t0));
    t0 = tc;

    Dataset<Row> dfGold = internationalPaxDf
        .join(domesticPaxDf,
            internationalPaxDf.col("month")
                .equalTo(domesticPaxDf.col("month")),
            "outer")
        .withColumn("pax", expr("internationalPax + domesticPax"))
        .drop(domesticPaxDf.col("month"))
        .filter(
            col("month").$less(lit("2020-01-01").cast(DataTypes.DateType)))
        .orderBy(col("month"))
        .cache();

    tc = System.currentTimeMillis();
    log.info("Transformation to gold zone in {} ms.", (tc - t0));
    t0 = tc;

    // Shows at most 5 rows from the dataframe
    dfGold.show(5, false);
    dfGold.printSchema();

    DrstiChart d = new DrstiLineChart(dfGold);
    d.render();

    tc = System.currentTimeMillis();
    log.info("Data exported for in {} ms.", (tc - t0));
    t0 = tc;

    Dataset<Row> dfQuarter = dfGold
        .withColumn("year", year(col("month")))
        .withColumn("q", ceil(month(col("month")).$div(3)))
        .withColumn("period", concat(col("year"), lit("-Q"), col("q")))
        .groupBy(col("period"))
        .agg(sum("pax").as("pax"),
            sum("internationalPax").as("internationalPax"),
            sum("domesticPax").as("domesticPax"))
        .drop("year")
        .drop("q")
        .orderBy(col("period"));

    dfQuarter.show(5, false);
    dfQuarter.printSchema();

    d = new DrstiLineChart(dfQuarter);
    d.render();

    Dataset<Row> dfYear = dfGold
        .withColumn("year", year(col("month")))
        .groupBy(col("year"))
        .agg(sum("pax").as("pax"),
            sum("internationalPax").as("internationalPax"),
            sum("domesticPax").as("domesticPax"))
        .orderBy(col("year"));
    dfYear = DrstiUtils.setHeader(dfYear, "year", "Year");
    dfYear = DrstiUtils.setHeader(dfYear, "pax", "Passengers");
    dfYear = DrstiUtils.setHeader(
        dfYear, "internationalPax", "International Passengers");
    dfYear = DrstiUtils.setHeader(
        dfYear, "domesticPax", "Domestic Passengers");
    
    dfYear.show(5, false);
    dfYear.printSchema();

    d = new DrstiLineChart(dfYear);
    d.setTitle("Air passenger traffic per year");
    d.setXTitle("Year " + DataframeUtils.min(dfYear, "year")+ " - "+ DataframeUtils.max(dfYear, "year"));
    d.setYTitle("Passenger (000s)");
    d.render();

    spark.stop();
    return true;
  }

}
