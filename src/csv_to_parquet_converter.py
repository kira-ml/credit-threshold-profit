"""
CSV to Parquet Converter for Lending Club Dataset
Optimized for Intel Core i5 laptop with memory constraints
Author: Production Quality Code
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import logging
from datetime import datetime
import gc
import sys
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('csv_to_parquet_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CSVToParquetConverter:
    """Optimized CSV to Parquet converter for large datasets on Core i5"""
    
    def __init__(self, input_path: str, output_dir: str, chunk_size: int = 50000):
        """
        Initialize converter with optimized settings for Core i5
        
        Args:
            input_path: Path to input CSV file
            output_dir: Directory for output Parquet files
            chunk_size: Number of rows per chunk (adjusted for i5 memory)
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        
        # Memory-optimized dtypes mapping
        self.dtype_mapping = {
            'int64': 'int32',  # Downcast integers where possible
            'float64': 'float32',  # Downcast floats
            'object': 'string',  # Use pandas string type for efficiency
        }
        
        # Columns that should remain as float64 (high precision needed)
        self.high_precision_cols = {
            'annual_inc', 'installment', 'dti', 'revol_bal', 'total_pymnt',
            'recoveries', 'collection_recovery_fee', 'int_rate'
        }
        
        # Date columns to parse
        self.date_columns = [
            'issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d',
            'last_credit_pull_d', 'hardship_start_date', 'hardship_end_date',
            'payment_plan_start_date', 'debt_settlement_flag_date', 'settlement_date'
        ]
        
        # Boolean flag columns (convert to boolean type)
        self.bool_columns = [
            'hardship_flag', 'debt_settlement_flag'
        ]
        
    def create_output_directory(self) -> None:
        """Create output directory if it doesn't exist"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created/verified: {self.output_dir}")
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame dtypes for memory efficiency
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with optimized dtypes
        """
        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        for col in df.columns:
            col_dtype = df[col].dtype
            
            # Skip high precision columns
            if col in self.high_precision_cols:
                continue
                
            # Convert boolean columns
            if col in self.bool_columns:
                df[col] = df[col].map({'Yes': True, 'No': False, '': None}).astype('boolean')
                continue
                
            
            # Convert string columns
            if col_dtype == 'object':
                # Check if column has limited unique values
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5 and df[col].nunique() < 100:
                    df[col] = df[col].astype('category')
                else:
                    df[col] = df[col].astype('string')
                continue
                
            # Optimize numeric columns
            if col_dtype == 'int64':
                # Check if column has only non-negative integers
                if df[col].min() >= 0:
                    # Try downcasting to smallest possible unsigned int
                    max_val = df[col].max()
                    if max_val <= 255:
                        df[col] = df[col].astype('uint8')
                    elif max_val <= 65535:
                        df[col] = df[col].astype('uint16')
                    elif max_val <= 4294967295:
                        df[col] = df[col].astype('uint32')
                else:
                    df[col] = df[col].astype('int32')
                    
            elif col_dtype == 'float64':
                # Check if column can be downcast to float32 without precision loss
                if df[col].notna().any():
                    if (df[col].fillna(0).astype('float32') == df[col].fillna(0)).all():
                        df[col] = df[col].astype('float32')
        
        optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
        memory_saved = original_memory - optimized_memory
        memory_percent = (memory_saved / original_memory) * 100
        
        logger.info(f"Memory optimization: {original_memory:.2f}MB -> {optimized_memory:.2f}MB "
                   f"(Saved: {memory_saved:.2f}MB / {memory_percent:.1f}%)")
        
        return df
    
    def parse_dates_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse date columns efficiently
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with parsed dates
        """
        for date_col in self.date_columns:
            if date_col in df.columns:
                try:
                    # Use pandas' optimized date parsing
                    df[date_col] = pd.to_datetime(
                        df[date_col], 
                        format='%b-%Y',  # Format like "Dec-2011"
                        errors='coerce'
                    )
                except Exception as e:
                    logger.warning(f"Could not parse column {date_col}: {e}")
        return df
    
    def process_chunk(self, chunk: pd.DataFrame, chunk_num: int) -> Optional[pd.DataFrame]:
        """
        Process a single chunk of data
        
        Args:
            chunk: DataFrame chunk
            chunk_num: Chunk number for logging
            
        Returns:
            Processed DataFrame or None if error
        """
        try:
            logger.info(f"Processing chunk {chunk_num} with {len(chunk)} rows")
            
            # Parse dates
            chunk = self.parse_dates_optimized(chunk)
            
            # Replace problematic values
            chunk = chunk.replace([np.inf, -np.inf], np.nan)
            
            # Optimize dtypes
            chunk = self.optimize_dtypes(chunk)
            
            logger.info(f"Chunk {chunk_num} processed successfully")
            return chunk
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_num}: {e}")
            return None
    
    def write_to_parquet(self, chunk: pd.DataFrame, chunk_num: int) -> None:
        """
        Write chunk to Parquet file
        
        Args:
            chunk: DataFrame to write
            chunk_num: Chunk number for filename
        """
        try:
            output_file = self.output_dir / f"chunk_{chunk_num:04d}.parquet"
            
            # Convert to PyArrow Table for better compression
            table = pa.Table.from_pandas(chunk)
            
            # Write with optimal compression for Core i5
            pq.write_table(
                table,
                output_file,
                compression='snappy',  # Good balance of speed/compression
                use_dictionary=True,
                write_statistics=True,
                data_page_size=1048576,  # 1MB pages for better I/O
            )
            
            logger.info(f"Written: {output_file} ({output_file.stat().st_size / 1024**2:.2f}MB)")
            
        except Exception as e:
            logger.error(f"Error writing chunk {chunk_num}: {e}")
            raise
    
    def convert_single_parquet(self) -> None:
        """
        Convert entire dataset to a single optimized Parquet file
        Use when dataset fits in memory after optimization
        """
        logger.info("Starting conversion to single Parquet file")
        
        try:
            # Read CSV with optimized parameters
            logger.info(f"Reading CSV file: {self.input_path}")
            df = pd.read_csv(
                self.input_path,
                low_memory=False,
                na_values=['', 'NA', 'NULL', 'NaN', 'null'],
                keep_default_na=True
            )
            
            logger.info(f"Initial rows: {len(df)}")
            
            # Process full DataFrame
            df = self.parse_dates_optimized(df)
            df = df.replace([np.inf, -np.inf], np.nan)
            df = self.optimize_dtypes(df)
            
            # Write to single Parquet file
            output_file = self.output_dir / "accepted_loans.parquet"
            table = pa.Table.from_pandas(df)
            
            pq.write_table(
                table,
                output_file,
                compression='snappy',
                use_dictionary=True,
                write_statistics=True,
                data_page_size=1048576,
            )
            
            file_size_mb = output_file.stat().st_size / 1024**2
            logger.info(f"Conversion complete! Output: {output_file} ({file_size_mb:.2f}MB)")
            
        except MemoryError:
            logger.warning("MemoryError: Falling back to chunked conversion")
            self.convert_chunked_parquet()
        except Exception as e:
            logger.error(f"Error in single conversion: {e}")
            raise
    
    def convert_chunked_parquet(self) -> None:
        """
        Convert CSV to multiple Parquet files using chunking
        Optimized for memory-constrained Core i5
        """
        logger.info(f"Starting chunked conversion with chunk size: {self.chunk_size}")
        
        # Count total rows (optional, for progress tracking)
        try:
            total_rows = sum(1 for _ in open(self.input_path)) - 1
            total_chunks = (total_rows + self.chunk_size - 1) // self.chunk_size
            logger.info(f"Estimated total rows: {total_rows:,}, chunks: {total_chunks}")
        except:
            total_chunks = None
        
        chunk_num = 0
        chunk_iter = pd.read_csv(
            self.input_path,
            chunksize=self.chunk_size,
            low_memory=False,
            na_values=['', 'NA', 'NULL', 'NaN', 'null'],
            keep_default_na=True
        )
        
        for chunk in chunk_iter:
            processed_chunk = self.process_chunk(chunk, chunk_num)
            if processed_chunk is not None:
                self.write_to_parquet(processed_chunk, chunk_num)
            
            # Force garbage collection
            del chunk
            del processed_chunk
            gc.collect()
            
            chunk_num += 1
            if chunk_num % 10 == 0:
                logger.info(f"Progress: Completed {chunk_num} chunks")
        
        logger.info(f"Chunked conversion complete! Created {chunk_num} Parquet files")
        
        # Create metadata file for chunks
        self.create_metadata_file(chunk_num)
    
    def create_metadata_file(self, num_chunks: int) -> None:
        """
        Create metadata file describing the chunked dataset
        
        Args:
            num_chunks: Number of chunks created
        """
        metadata = {
            'dataset': 'Lending Club Accepted Loans',
            'source_file': str(self.input_path),
            'num_chunks': num_chunks,
            'chunk_size': self.chunk_size,
            'conversion_date': datetime.now().isoformat(),
            'notes': 'Dataset split into multiple Parquet files for memory efficiency'
        }
        
        import json
        metadata_file = self.output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata written to: {metadata_file}")
    
    def read_parquet_dataset(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Helper method to read back the converted Parquet dataset
        
        Args:
            columns: Specific columns to read (optional)
            
        Returns:
            Combined DataFrame from all Parquet files
        """
        logger.info("Reading back Parquet dataset")
        
        parquet_files = sorted(self.output_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found in {self.output_dir}")
        
        # Read all parquet files
        dfs = []
        for pq_file in parquet_files:
            logger.info(f"Reading: {pq_file}")
            if columns:
                df = pd.read_parquet(pq_file, columns=columns)
            else:
                df = pd.read_parquet(pq_file)
            dfs.append(df)
        
        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined_df):,} rows from {len(parquet_files)} files")
        
        return combined_df


def main():
    """Main execution function"""
    
    # Configuration
    INPUT_CSV = r"D:\archive (28)\accepted_2007_to_2018q4.csv\accepted_2007_to_2018Q4.csv"
    OUTPUT_DIR = r"D:\credit-threshold-profit\data\raw"
    
    # Initialize converter
    # Using smaller chunk size for Core i5 (adjust based on your RAM)
    converter = CSVToParquetConverter(
        input_path=INPUT_CSV,
        output_dir=OUTPUT_DIR,
        chunk_size=50000  # Adjust: 50k rows per chunk ~500MB-1GB memory
    )
    
    try:
        # Create output directory
        converter.create_output_directory()
        
        # Check if input file exists
        if not converter.input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {converter.input_path}")
        
        # Get file size for logging
        file_size_gb = converter.input_path.stat().st_size / 1024**3
        logger.info(f"Input file size: {file_size_gb:.2f} GB")
        
        # Choose conversion strategy based on file size
        # For files > 2GB on Core i5, use chunked conversion
        if file_size_gb > 2.0:
            logger.info("Large file detected, using chunked conversion")
            converter.convert_chunked_parquet()
        else:
            logger.info("Attempting single file conversion")
            converter.convert_single_parquet()
            
        logger.info("Conversion completed successfully!")
        
        # Optional: Test reading back a sample
        # sample_df = converter.read_parquet_dataset(columns=['id', 'loan_amnt', 'int_rate'])
        # logger.info(f"Sample data shape: {sample_df.shape}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()