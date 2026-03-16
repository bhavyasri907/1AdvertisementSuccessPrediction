# enhanced_video_analyzer.py
import cv2
import numpy as np
from collections import Counter
import tempfile
import os

class VideoAnalyzer:
    def __init__(self):
        # Initialize OpenCV detectors
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
    def analyze_ad_video(self, video_path: str, ml_rating: float, ml_success_prob: float, ml_money_pred: str) -> str:
        """
        Enhanced video analysis with comprehensive metrics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return "Error: Could not open video file for analysis."

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Metrics tracking
        metrics = {
            'brightness': [],
            'contrast': [],
            'scene_cuts': 0,
            'faces': 0,
            'smiles': 0,
            'motion': [],
            'colorfulness': [],
            'text_regions': 0,
            'dominant_colors': [],
            'frame_with_logo': 0
        }
        
        prev_frame = None
        prev_hist = None
        frames_processed = 0
        
        # Process frames at reduced rate for efficiency
        frame_skip = max(1, int(fps / 10))  # Process 10 fps
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            if current_frame % frame_skip != 0:
                continue
                
            frames_processed += 1
            
            # Convert to different color spaces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 1. Brightness and contrast
            metrics['brightness'].append(gray.mean())
            metrics['contrast'].append(gray.std())
            
            # 2. Color analysis
            color_std = hsv.std(axis=(0,1)).mean()
            metrics['colorfulness'].append(color_std)
            
            # Get dominant color (simplified - using average HSV)
            dominant_hue = hsv[..., 0].mean()
            metrics['dominant_colors'].append(dominant_hue)
            
            # 3. Motion detection
            if prev_frame is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                motion_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
                metrics['motion'].append(motion_magnitude)
            prev_frame = gray.copy()
            
            # 4. Scene cut detection
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist)
            
            if prev_hist is not None:
                hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if hist_diff > 0.4:  # Threshold for scene cut
                    metrics['scene_cuts'] += 1
            prev_hist = hist
            
            # 5. Face and expression detection (every 15th frame)
            if frames_processed % 15 == 0:
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    metrics['faces'] += 1
                    
                    # Check for smile in face region
                    face_roi = gray[y:y+h, x:x+w]
                    smiles = self.smile_cascade.detectMultiScale(face_roi, 1.8, 20)
                    metrics['smiles'] += len(smiles)
            
            # 6. Text/logo detection (edge density heuristic)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            if edge_density > 0.15:  # High edge density might indicate text/logo
                metrics['text_regions'] += 1

        cap.release()

        # Calculate averages
        if frames_processed > 0:
            avg_brightness = np.mean(metrics['brightness'])
            avg_contrast = np.mean(metrics['contrast'])
            avg_motion = np.mean(metrics['motion']) if metrics['motion'] else 0
            avg_colorfulness = np.mean(metrics['colorfulness'])
            text_percentage = (metrics['text_regions'] / frames_processed) * 100
            
            # Calculate shot length (average duration between cuts)
            if metrics['scene_cuts'] > 0:
                avg_shot_length = duration / (metrics['scene_cuts'] + 1)
            else:
                avg_shot_length = duration
        else:
            avg_brightness = avg_contrast = avg_motion = avg_colorfulness = text_percentage = 0
            avg_shot_length = duration

        # Generate comprehensive report
        report = self._generate_report(
            metrics, avg_brightness, avg_contrast, avg_motion, avg_colorfulness,
            text_percentage, avg_shot_length, duration, frames_processed,
            ml_rating, ml_success_prob, ml_money_pred
        )
        
        return report
    
    def _generate_report(self, metrics, brightness, contrast, motion, colorfulness,
                        text_pct, shot_length, duration, frames_processed,
                        ml_rating, ml_prob, ml_money):
        """Generate detailed marketing insights"""
        
        # Classifications
        brightness_label = "High (Vibrant)" if brightness > 130 else "Medium" if brightness > 90 else "Low (Dark)"
        contrast_label = "High (Dramatic)" if contrast > 60 else "Medium" if contrast > 40 else "Low (Flat)"
        motion_label = "Dynamic" if motion > 5 else "Static" if motion < 2 else "Moderate"
        color_label = "Vibrant" if colorfulness > 50 else "Muted" if colorfulness < 30 else "Balanced"
        
        # Pacing analysis
        if shot_length < 2:
            pacing = "Very Fast (Action-packed)"
            pacing_score = 9
        elif shot_length < 4:
            pacing = "Fast (Dynamic)"
            pacing_score = 7
        elif shot_length < 7:
            pacing = "Moderate (Standard)"
            pacing_score = 5
        elif shot_length < 12:
            pacing = "Slow (Thoughtful)"
            pacing_score = 3
        else:
            pacing = "Very Slow (Continuous)"
            pacing_score = 1
            
        # Face analysis
        face_density = metrics['faces'] / (duration * (frames_processed / 30)) if duration > 0 else 0
        if face_density > 0.5:
            human_element = "Strong human presence (✅ Good for engagement)"
            human_score = 9
        elif face_density > 0.2:
            human_element = "Moderate human presence"
            human_score = 6
        else:
            human_element = "Minimal human presence (Product-focused)"
            human_score = 3
            
        # Smile ratio
        if metrics['faces'] > 0:
            smile_ratio = metrics['smiles'] / metrics['faces']
            if smile_ratio > 0.3:
                emotion = "Positive (Smiles detected ✅)"
            elif smile_ratio > 0.1:
                emotion = "Neutral"
            else:
                emotion = "Serious/Professional"
        else:
            emotion = "No faces detected"
            smile_ratio = 0
            
        # Scene cut analysis
        cuts_per_minute = (metrics['scene_cuts'] / duration) * 60 if duration > 0 else 0
        
        # Text analysis
        if text_pct > 30:
            text_strategy = "Heavy text usage (May overwhelm)"
        elif text_pct > 10:
            text_strategy = "Balanced text overlay"
        else:
            text_strategy = "Minimal text (Visual-focused)"
            
        # Calculate engagement score (0-100)
        engagement_score = (
            min(brightness / 2.55, 25) +  # 25% weight
            min(contrast / 2.4, 25) +      # 25% weight
            min(pacing_score * 3, 25) +     # 25% weight
            min(human_score * 3, 25)        # 25% weight
        )
        
        # ML-CV alignment
        if ml_prob > 70:
            alignment = "✅ Strong alignment with ML prediction"
            confidence = "High"
        elif ml_prob > 50:
            alignment = "⚠️ Moderate alignment - room for improvement"
            confidence = "Medium"
        else:
            alignment = "❌ Misalignment - consider creative revision"
            confidence = "Low"
            
        # Generate report
        report = f"""
## 🎯 Video Analysis Report

### 📊 Visual Metrics
| Metric | Value | Score |
|--------|-------|-------|
| Brightness | {brightness_label} ({brightness:.1f}/255) | {min(brightness/2.55, 25):.1f}/25 |
| Contrast | {contrast_label} ({contrast:.1f}) | {min(contrast/2.4, 25):.1f}/25 |
| Color Vibrancy | {color_label} | {min(colorfulness/2, 25):.1f}/25 |
| Motion | {motion_label} | {min(motion*5, 25):.1f}/25 |
| Pacing | {pacing} ({cuts_per_minute:.1f} cuts/min) | {pacing_score}/10 |
| Text Elements | {text_strategy} | - |

### 👥 Human Elements
- **Human Presence:** {human_element}
- **Faces Detected:** {metrics['faces']}
- **Emotional Tone:** {emotion} (Smile ratio: {smile_ratio:.2f})
- **Face Density:** {face_density:.2f} faces/second

### 🎨 Creative Analysis
- **Engagement Score:** {engagement_score:.1f}/100
- **Shot Length:** {shot_length:.1f} seconds average
- **Color Palette:** {self._get_color_description(metrics['dominant_colors'])}
- **Visual Complexity:** {self._get_complexity_description(colorfulness, text_pct)}

### 📈 ML-CV Synergy
- **ML Success Probability:** {ml_prob:.1f}%
- **ML Rating Prediction:** {ml_rating:.2f}/5
- **Money-back Guarantee:** {ml_money}
- **Alignment:** {alignment}
- **Prediction Confidence:** {confidence}

### 💡 Actionable Recommendations
{self._get_recommendations(brightness, contrast, motion, pacing_score, 
                           human_score, text_pct, ml_prob, ml_money)}

---
*Analysis generated locally using OpenCV - no data leaves your device*
"""
        return report
    
    def _get_color_description(self, hues):
        """Describe dominant colors based on hue values"""
        if not hues:
            return "Unknown"
        avg_hue = np.mean(hues)
        if avg_hue < 30 or avg_hue > 330:
            return "Warm (Reds/Oranges)"
        elif avg_hue < 90:
            return "Warm (Yellows)"
        elif avg_hue < 150:
            return "Cool (Greens)"
        elif avg_hue < 210:
            return "Cool (Cyans)"
        elif avg_hue < 270:
            return "Cool (Blues)"
        elif avg_hue < 330:
            return "Cool (Purples)"
        return "Mixed"
    
    def _get_complexity_description(self, colorfulness, text_pct):
        """Describe visual complexity"""
        if colorfulness > 50 and text_pct > 30:
            return "High (Busy/Distracting)"
        elif colorfulness > 50 or text_pct > 30:
            return "Medium"
        else:
            return "Low (Clean/Simple)"
    
    def _get_recommendations(self, brightness, contrast, motion, pacing_score,
                            human_score, text_pct, ml_prob, ml_money):
        """Generate actionable recommendations"""
        recs = []
        
        # Visual recommendations
        if brightness < 90:
            recs.append("🔆 **Increase brightness** - Current lighting is too dark for optimal engagement")
        elif brightness > 200:
            recs.append("🌙 **Reduce brightness** - Slightly overexposed, consider dimming")
        else:
            recs.append("✅ **Good lighting** - Brightness is well-optimized")
            
        if contrast < 40:
            recs.append("🎨 **Boost contrast** - Add more visual punch to make elements stand out")
            
        # Pacing recommendations
        if pacing_score < 4:
            if ml_prob < 60:
                recs.append("⚡ **Increase pacing** - Shorter shots (2-4 seconds) typically perform better")
            else:
                recs.append("⏱️ **Consider pacing** - Current slow pace works but test faster versions")
        elif pacing_score > 7:
            if ml_prob < 60:
                recs.append("🎬 **Slightly reduce pacing** - Very fast cuts may overwhelm viewers")
                
        # Human element recommendations
        if human_score < 4:
            recs.append("👥 **Add human elements** - Faces increase emotional connection and trust")
        elif human_score > 7:
            recs.append("😊 **Leverage humans** - Feature faces prominently in thumbnails/previews")
            
        # Text recommendations
        if text_pct > 30:
            recs.append("📝 **Simplify text** - Too much text can distract from the message")
        elif text_pct < 10 and ml_prob < 50:
            recs.append("📋 **Add key text** - Important benefits could be highlighted with text overlays")
            
        # ML-based recommendations
        if ml_prob > 80:
            recs.append("🚀 **High potential ad** - Ready for production, consider A/B testing")
        elif ml_prob > 60:
            recs.append("📈 **Promising ad** - Test with small budget before scaling")
        else:
            recs.append("🔄 **Needs revision** - Consider major creative changes based on feedback")
            
        # Money-back guarantee
        if ml_money == "Yes":
            recs.append("💰 **Money-back guarantee** - Highlight this offer in the ad")
        else:
            recs.append("💡 **Consider guarantee** - Adding a guarantee might boost conversions")
            
        return "\n".join(recs)