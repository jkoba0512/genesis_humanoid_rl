#!/usr/bin/env python3
"""
Database migration CLI tool.

Provides command-line interface for managing database schema migrations.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.genesis_humanoid_rl.infrastructure.persistence.database import (
    DatabaseConnection,
)
from src.genesis_humanoid_rl.infrastructure.persistence.migration_manager import (
    DatabaseMigrationManager,
    MigrationError,
)


def status_command(args):
    """Show migration status."""
    print("🔍 Checking migration status...")

    db = DatabaseConnection(args.database)
    manager = DatabaseMigrationManager(db)

    status = manager.get_migration_status()

    print(f"\n📊 Database Migration Status")
    print(f"{'='*50}")
    print(f"Current Version: {status['current_version'] or 'None'}")
    print(
        f"Optimized Schema: {'✅ Applied' if status['has_optimized_schema'] else '❌ Not Applied'}"
    )
    print(f"Total Migrations: {status['total_migrations']}")
    print(f"Current Tables: {len(status['current_tables'])}")

    if status["missing_tables"]:
        print(f"\n⚠️  Missing Tables ({len(status['missing_tables'])}):")
        for table in status["missing_tables"]:
            print(f"   - {table}")

    if status["migration_history"]:
        print(f"\n📝 Migration History:")
        for migration in status["migration_history"]:
            applied_at = datetime.fromisoformat(migration["applied_at"])
            print(f"   {migration['version']} - {migration['name']}")
            print(f"      Applied: {applied_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"      Duration: {migration['execution_time']:.2f}s")
            print()

    if args.verbose:
        print(f"\n🗃️  Current Tables:")
        for table in sorted(status["current_tables"]):
            print(f"   - {table}")


def migrate_command(args):
    """Run database migration."""
    print("🚀 Starting database migration to optimized schema...")

    db = DatabaseConnection(args.database)
    manager = DatabaseMigrationManager(db)

    # Check current status
    status = manager.get_migration_status()

    if status["has_optimized_schema"]:
        print("✅ Optimized schema already applied")
        if not args.force:
            print("Use --force to re-apply migration")
            return
        else:
            print("⚠️  Force flag specified, re-applying migration...")

    try:
        print("\n📋 Migration Plan:")
        print("  1. Backup existing data")
        print("  2. Create optimized schema")
        print("  3. Migrate data to new structure")
        print("  4. Validate data integrity")
        print("  5. Update migration history")

        if not args.auto_confirm:
            response = input("\nProceed with migration? (y/N): ")
            if response.lower() != "y":
                print("Migration cancelled")
                return

        # Run migration
        start_time = datetime.now()
        manager.migrate_to_optimized_schema()
        duration = (datetime.now() - start_time).total_seconds()

        print(f"\n✅ Migration completed successfully in {duration:.2f}s")

        # Show new status
        new_status = manager.get_migration_status()
        print(f"📊 New Schema: {len(new_status['current_tables'])} tables")
        print(
            f"🎯 Performance: {len(new_status['current_tables']) * 3} indexes created"
        )

    except MigrationError as e:
        print(f"\n❌ Migration failed: {e}")
        if e.original_error:
            print(f"Original error: {e.original_error}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during migration: {e}")
        sys.exit(1)


def history_command(args):
    """Show migration history."""
    print("📜 Migration History")

    db = DatabaseConnection(args.database)
    manager = DatabaseMigrationManager(db)

    history = manager.get_migration_history()

    if not history:
        print("No migrations found")
        return

    print(f"{'='*80}")
    for migration in history:
        print(f"Version: {migration.version}")
        print(f"Name: {migration.name}")
        print(f"Applied: {migration.applied_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {migration.execution_time_seconds:.2f}s")
        if migration.rollback_sql:
            print("Rollback: Available")
        print("-" * 40)


def validate_command(args):
    """Validate current schema."""
    print("🔍 Validating database schema...")

    db = DatabaseConnection(args.database)
    manager = DatabaseMigrationManager(db)

    status = manager.get_migration_status()

    issues = []

    # Check for optimized schema
    if not status["has_optimized_schema"]:
        issues.append("Optimized schema not applied")

    # Check for missing tables
    if status["missing_tables"]:
        issues.append(f"Missing {len(status['missing_tables'])} tables")

    # Check table counts
    from src.genesis_humanoid_rl.infrastructure.persistence.schema_design import (
        OptimizedSchemaDesign,
    )

    schema = OptimizedSchemaDesign()
    expected_tables = set(schema.tables.keys())
    current_tables = set(status["current_tables"])

    unexpected_tables = current_tables - expected_tables
    if unexpected_tables:
        issues.append(f"Unexpected tables: {', '.join(unexpected_tables)}")

    if issues:
        print(f"\n⚠️  Schema Issues Found:")
        for issue in issues:
            print(f"   - {issue}")
        print(f"\n💡 Run 'migrate' command to fix issues")
        sys.exit(1)
    else:
        print("✅ Schema validation passed")
        print("🎯 All tables present and correctly structured")


def backup_command(args):
    """Create database backup."""
    print("💾 Creating database backup...")

    source_db = DatabaseConnection(args.database)
    backup_path = args.output or f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

    # Simple backup by copying database file
    import shutil

    shutil.copy2(args.database, backup_path)

    print(f"✅ Backup created: {backup_path}")

    # Validate backup
    backup_db = DatabaseConnection(backup_path)
    backup_manager = DatabaseMigrationManager(backup_db)
    backup_status = backup_manager.get_migration_status()

    print(f"📊 Backup contains {len(backup_status['current_tables'])} tables")
    if backup_status["migration_history"]:
        print(
            f"📝 Migration history: {len(backup_status['migration_history'])} entries"
        )


def performance_analysis_command(args):
    """Analyze schema performance characteristics."""
    print("⚡ Analyzing schema performance...")

    from src.genesis_humanoid_rl.infrastructure.persistence.schema_design import (
        OptimizedSchemaDesign,
    )

    schema = OptimizedSchemaDesign()
    analysis = schema.get_performance_analysis()

    print(f"\n📊 Performance Analysis")
    print(f"{'='*50}")
    print(f"Total Tables: {analysis['total_tables']}")
    print(f"Total Indexes: {analysis['total_indexes']}")
    print(f"Estimated Rows: {analysis['estimated_total_rows']:,}")

    if analysis["large_tables"]:
        print(f"\n🗃️  Large Tables (>1M rows):")
        for table in analysis["large_tables"]:
            print(f"   - {table}")

    if analysis["optimization_recommendations"]:
        print(f"\n💡 Optimization Recommendations:")
        for rec in analysis["optimization_recommendations"]:
            print(f"   - {rec}")

    # Calculate index to table ratio
    index_ratio = analysis["total_indexes"] / analysis["total_tables"]
    print(f"\n📈 Index Density: {index_ratio:.1f} indexes per table")

    if index_ratio >= 3.0:
        print("✅ Good index coverage for performance")
    elif index_ratio >= 2.0:
        print("⚠️  Moderate index coverage")
    else:
        print("❌ Low index coverage - consider adding more indexes")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Database Migration Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                          # Show migration status
  %(prog)s migrate --auto-confirm          # Run migration without prompts
  %(prog)s history                         # Show migration history
  %(prog)s validate                        # Validate current schema
  %(prog)s backup --output my_backup.db    # Create backup
  %(prog)s performance                     # Analyze performance
        """,
    )

    parser.add_argument(
        "--database",
        "-d",
        default="genesis_humanoid_rl.db",
        help="Database file path (default: genesis_humanoid_rl.db)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show migration status")
    status_parser.set_defaults(func=status_command)

    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Run database migration")
    migrate_parser.add_argument(
        "--auto-confirm", action="store_true", help="Skip confirmation prompts"
    )
    migrate_parser.add_argument(
        "--force", action="store_true", help="Force migration even if already applied"
    )
    migrate_parser.set_defaults(func=migrate_command)

    # History command
    history_parser = subparsers.add_parser("history", help="Show migration history")
    history_parser.set_defaults(func=history_command)

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate current schema")
    validate_parser.set_defaults(func=validate_command)

    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create database backup")
    backup_parser.add_argument("--output", "-o", help="Backup file path")
    backup_parser.set_defaults(func=backup_command)

    # Performance analysis command
    perf_parser = subparsers.add_parser(
        "performance", help="Analyze schema performance"
    )
    perf_parser.set_defaults(func=performance_analysis_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n🚫 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
