#!/usr/bin/env python3
"""
VideoVault - Intelligent Metadata Management UI for ART.WE.ED.IT
Built with Gradio for real-time video classification and bulk tagging
"""

import os
import json
import gradio as gr
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoVaultUI:
    """Interactive video metadata management interface."""
    
    PREDEFINED_TAGS = [
        'action', 'combat', 'chase', 'explosion', 'character_focus',
        'visual_quality', 'slow_motion', 'fast_cut', 'smoke', 'neon',
        'dramatic', 'transition', 'dance', 'silhouette', 'closeup'
    ]
    
    SHOT_SIZES = [
        'extreme_closeup', 'closeup', 'medium_shot', 
        'full_shot', 'wide_shot', 'extreme_wide'
    ]
    
    CAMERA_MOTIONS = [
        'static', 'pan', 'tilt', 'zoom', 'dolly',
        'handheld', 'tracking', 'crane', 'drone'
    ]
    
    LIGHTING_TYPES = [
        'lowkey', 'highkey', 'neon', 'backlighting', 
        'sidelighting', 'practical', 'color_shift', 'strobe'
    ]
    
    def __init__(self, cache_dir: str = "input/video_analysis_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.videos = self._load_videos()
        self.current_selection = []
        logger.info(f"VideoVault initialized with {len(self.videos)} cached videos")
    
    def _load_videos(self) -> List[Dict]:
        """Load all cached video metadata."""
        videos = []
        
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith("_metadata.json"):
                    try:
                        with open(os.path.join(self.cache_dir, filename)) as f:
                            metadata = json.load(f)
                            metadata['_filename'] = filename
                            videos.append(metadata)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse {filename}")
        
        return videos
    
    def get_video_dropdown_choices(self) -> List[str]:
        """Generate dropdown choices for video selector."""
        return [
            f"[{i}] {v.get('filename', v.get('_filename', 'Unknown'))}" 
            for i, v in enumerate(self.videos)
        ]
    
    def parse_video_index(self, choice: str) -> int:
        """Extract video index from dropdown choice."""
        try:
            return int(choice.split(']')[0].strip('['))
        except (ValueError, IndexError):
            return 0
    
    # ========================================================================
    # TAB 1: INDIVIDUAL TAGGING
    # ========================================================================
    
    def on_video_select(self, video_choice: str) -> Tuple[str, str, str, str]:
        """Handle video selection and display metadata."""
        if not video_choice or not self.videos:
            return "", "", "", ""
        
        idx = self.parse_video_index(video_choice)
        if idx >= len(self.videos):
            return "", "", "", ""
        
        video = self.videos[idx]
        
        # Extract metadata
        semantic = video.get('semantic', {})
        camera_motion = semantic.get('camera_motion', 'static')
        lighting = semantic.get('lighting', 'natural')
        shot_size = semantic.get('shot_size', 'medium_shot')
        emotional_tags = ', '.join(semantic.get('emotional_tags', []))
        
        return emotional_tags, camera_motion, lighting, shot_size
    
    def save_individual_tags(self, video_choice: str, tags: str, 
                            camera: str, lighting: str, shot: str) -> str:
        """Save manually edited tags for a single video."""
        if not video_choice or not self.videos:
            return "❌ No video selected"
        
        idx = self.parse_video_index(video_choice)
        if idx >= len(self.videos):
            return "❌ Invalid video index"
        
        video = self.videos[idx]
        
        # Update semantic metadata
        if 'semantic' not in video:
            video['semantic'] = {}
        
        video['semantic']['emotional_tags'] = [t.strip() for t in tags.split(',')]
        video['semantic']['camera_motion'] = camera
        video['semantic']['lighting'] = lighting
        video['semantic']['shot_size'] = shot
        
        # Save to file
        try:
            cache_file = os.path.join(self.cache_dir, video['_filename'])
            with open(cache_file, 'w') as f:
                json.dump(video, f, indent=2)
            
            logger.info(f"Saved metadata for {video['_filename']}")
            return f"✅ Saved tags for video {idx}"
        
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return f"❌ Error: {str(e)}"
    
    # ========================================================================
    # TAB 2: BULK EDITING
    # ========================================================================
    
    def parse_indices(self, indices_str: str) -> List[int]:
        """Parse comma-separated indices."""
        try:
            return [int(x.strip()) for x in indices_str.split(',') if x.strip()]
        except ValueError:
            return []
    
    def bulk_edit_metadata(self, indices_str: str, tags: str,
                          shot_size: str, camera: str, 
                          lighting: str) -> str:
        """Apply bulk metadata changes."""
        
        indices = self.parse_indices(indices_str)
        if not indices:
            return "❌ No valid indices provided"
        
        updated_count = 0
        failed_videos = []
        
        for idx in indices:
            if idx >= len(self.videos):
                failed_videos.append(str(idx))
                continue
            
            video = self.videos[idx]
            
            # Update metadata
            if 'semantic' not in video:
                video['semantic'] = {}
            
            if tags:
                video['semantic']['emotional_tags'] = [
                    t.strip() for t in tags.split(',')
                ]
            
            if shot_size:
                video['semantic']['shot_size'] = shot_size
            
            if camera:
                video['semantic']['camera_motion'] = camera
            
            if lighting:
                video['semantic']['lighting'] = lighting
            
            # Save
            try:
                cache_file = os.path.join(self.cache_dir, video['_filename'])
                with open(cache_file, 'w') as f:
                    json.dump(video, f, indent=2)
                
                updated_count += 1
            except Exception as e:
                failed_videos.append(str(idx))
                logger.error(f"Failed to save video {idx}: {e}")
        
        result = f"✅ Updated {updated_count} videos"
        if failed_videos:
            result += f"\n⚠️ Failed: {', '.join(failed_videos)}"
        
        return result
    
    # ========================================================================
    # TAB 3: BATCH EXPORT & REPORTING
    # ========================================================================
    
    def generate_metadata_report(self) -> str:
        """Generate comprehensive metadata report."""
        
        report = "# Video Metadata Report\n\n"
        report += f"**Generated:** {Path(self.cache_dir).stat().st_mtime}\n"
        report += f"**Total Videos:** {len(self.videos)}\n\n"
        
        # Statistics
        tags_counter = {}
        shot_sizes = {}
        cameras = {}
        lighting_types = {}
        
        for video in self.videos:
            semantic = video.get('semantic', {})
            
            # Tags
            for tag in semantic.get('emotional_tags', []):
                tags_counter[tag] = tags_counter.get(tag, 0) + 1
            
            # Shot size
            shot = semantic.get('shot_size', 'unknown')
            shot_sizes[shot] = shot_sizes.get(shot, 0) + 1
            
            # Camera
            cam = semantic.get('camera_motion', 'unknown')
            cameras[cam] = cameras.get(cam, 0) + 1
            
            # Lighting
            light = semantic.get('lighting', 'unknown')
            lighting_types[light] = lighting_types.get(light, 0) + 1
        
        report += "## Tag Distribution\n"
        for tag, count in sorted(tags_counter.items(), key=lambda x: x[1], reverse=True):
            report += f"- {tag}: {count}\n"
        
        report += "\n## Shot Size Distribution\n"
        for shot, count in sorted(shot_sizes.items(), key=lambda x: x[1], reverse=True):
            report += f"- {shot}: {count}\n"
        
        report += "\n## Camera Motion Distribution\n"
        for cam, count in sorted(cameras.items(), key=lambda x: x[1], reverse=True):
            report += f"- {cam}: {count}\n"
        
        report += "\n## Lighting Distribution\n"
        for light, count in sorted(lighting_types.items(), key=lambda x: x[1], reverse=True):
            report += f"- {light}: {count}\n"
        
        return report
    
    def export_metadata_json(self) -> str:
        """Export all metadata as JSON."""
        try:
            export_path = os.path.join(self.cache_dir, "all_metadata_export.json")
            
            export_data = {
                'total_videos': len(self.videos),
                'export_timestamp': str(Path(self.cache_dir).stat().st_mtime),
                'videos': self.videos
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return f"✅ Exported {len(self.videos)} videos to {export_path}"
        
        except Exception as e:
            return f"❌ Export failed: {str(e)}"
    
    # ========================================================================
    # GRADIO INTERFACE
    # ========================================================================
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio UI."""
        
        with gr.Blocks(
            title="VideoVault - AI Video Metadata Management",
            theme=gr.themes.Soft(primary_hue="orange")
        ) as demo:
            
            gr.Markdown("""
            # 🎬 VideoVault
            ## Intelligent Video Metadata Management for ART.WE.ED.IT
            
            Manage, tag, and organize video clips with AI-powered semantic classification.
            """)
            
            with gr.Tabs():
                # ============================================================
                # TAB 1: INDIVIDUAL TAGGING
                # ============================================================
                with gr.Tab("👁️ Individual Tagging"):
                    gr.Markdown("### Manually Refine Video Metadata")
                    
                    with gr.Row():
                        video_selector = gr.Dropdown(
                            choices=self.get_video_dropdown_choices(),
                            label="Select Video",
                            interactive=True
                        )
                        refresh_btn = gr.Button("🔄 Refresh", size="sm")
                    
                    with gr.Row():
                        with gr.Column():
                            tags_input = gr.Textbox(
                                label="Emotional Tags (comma-separated)",
                                placeholder="action, dramatic, smoke",
                                lines=2
                            )
                        
                        with gr.Column():
                            camera_dropdown = gr.Dropdown(
                                choices=self.CAMERA_MOTIONS,
                                label="Camera Motion"
                            )
                            
                            lighting_dropdown = gr.Dropdown(
                                choices=self.LIGHTING_TYPES,
                                label="Lighting"
                            )
                            
                            shot_dropdown = gr.Dropdown(
                                choices=self.SHOT_SIZES,
                                label="Shot Size"
                            )
                    
                    save_btn = gr.Button("💾 Save Changes", variant="primary")
                    save_status = gr.Textbox(label="Status", interactive=False)
                    
                    # Callbacks
                    video_selector.change(
                        self.on_video_select,
                        inputs=video_selector,
                        outputs=[tags_input, camera_dropdown, lighting_dropdown, shot_dropdown]
                    )
                    
                    save_btn.click(
                        self.save_individual_tags,
                        inputs=[video_selector, tags_input, camera_dropdown, 
                               lighting_dropdown, shot_dropdown],
                        outputs=save_status
                    )
                    
                    refresh_btn.click(
                        lambda: (self._load_videos(), 
                                gr.Dropdown(choices=self.get_video_dropdown_choices())),
                        outputs=video_selector
                    )
                
                # ============================================================
                # TAB 2: BULK EDITING
                # ============================================================
                with gr.Tab("⚙️ Bulk Edit"):
                    gr.Markdown("### Apply Metadata Changes to Multiple Videos")
                    
                    with gr.Row():
                        indices_input = gr.Textbox(
                            label="Video Indices (comma-separated)",
                            placeholder="0,1,2,3",
                            lines=1
                        )
                    
                    with gr.Row():
                        with gr.Column():
                            bulk_tags = gr.Textbox(
                                label="Tags (leave blank to skip)",
                                placeholder="action,fast_cut,dramatic"
                            )
                        
                        with gr.Column():
                            bulk_shot_size = gr.Dropdown(
                                choices=[""] + self.SHOT_SIZES,
                                label="Override Shot Size"
                            )
                            
                            bulk_camera = gr.Dropdown(
                                choices=[""] + self.CAMERA_MOTIONS,
                                label="Override Camera Motion"
                            )
                            
                            bulk_lighting = gr.Dropdown(
                                choices=[""] + self.LIGHTING_TYPES,
                                label="Override Lighting"
                            )
                    
                    bulk_execute = gr.Button("🔄 Apply to All Selected", variant="primary")
                    bulk_status = gr.Textbox(label="Result", interactive=False)
                    
                    bulk_execute.click(
                        self.bulk_edit_metadata,
                        inputs=[indices_input, bulk_tags, bulk_shot_size, 
                               bulk_camera, bulk_lighting],
                        outputs=bulk_status
                    )
                
                # ============================================================
                # TAB 3: REPORTING & EXPORT
                # ============================================================
                with gr.Tab("📊 Reports & Export"):
                    gr.Markdown("### Generate Reports and Export Metadata")
                    
                    with gr.Row():
                        report_btn = gr.Button("📋 Generate Report", variant="primary")
                        export_btn = gr.Button("💾 Export JSON", variant="primary")
                    
                    report_output = gr.Markdown(label="Metadata Report")
                    
                    report_btn.click(
                        self.generate_metadata_report,
                        outputs=report_output
                    )
                    
                    export_btn.click(
                        self.export_metadata_json,
                        outputs=gr.Textbox(label="Export Status")
                    )
                
                # ============================================================
                # TAB 4: SETTINGS
                # ============================================================
                with gr.Tab("⚙️ Settings"):
                    gr.Markdown("### VideoVault Settings")
                    
                    cache_path_display = gr.Textbox(
                        value=self.cache_dir,
                        label="Cache Directory",
                        interactive=False
                    )
                    
                    stats_display = gr.Markdown(
                        f"""
                        ### Cache Statistics
                        - **Total Videos Cached:** {len(self.videos)}
                        - **Cache Path:** {self.cache_dir}
                        - **Tags Available:** {len(self.PREDEFINED_TAGS)}
                        """
                    )
        
        return demo


def main():
    """Launch VideoVault UI."""
    
    vault = VideoVaultUI()
    app = vault.create_interface()
    
    logger.info("🚀 Starting VideoVault UI")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
