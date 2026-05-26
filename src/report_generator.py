#!/usr/bin/env python3

import os
import tempfile
from collections import Counter
from datetime import datetime

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node

from rins_robot.msg import CylinderCoords, RingCoords
from rins_robot.srv import (
    GenerateReport,
    ReportAnomalyTile,
    ReportBarrelImage,
    ReportTaskAssignment,
)

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_HEADER_COLOR = colors.HexColor('#1a1a8c')
_ROW_COLORS = [colors.HexColor('#f0f0f0'), colors.white]


class ReportGenerator(Node):
    def __init__(self):
        super().__init__('report_generator')

        self.bridge = CvBridge()
        self._tmp_dir = tempfile.mkdtemp(prefix='rins_report_')

        # Accumulated data
        self.rings = []       # list of (id, color)
        self.cylinders = []   # list of (id, color, orientation, leaking)
        self.task_assignments = {}  # task -> {person, cell}
        self.anomaly_tiles = []     # list of dicts
        self.barrel_images = {}     # barrel_id -> img_path

        # Subscribers
        self.create_subscription(RingCoords, '/ring_coords', self._ring_cb, 10)
        self.create_subscription(CylinderCoords, '/cylinder_coords', self._cylinder_cb, 10)

        # Services provided by this node
        self.create_service(ReportTaskAssignment, '/report_task_assignment', self._task_assignment_cb)
        self.create_service(ReportAnomalyTile, '/report_anomaly_tile', self._anomaly_tile_cb)
        self.create_service(ReportBarrelImage, '/report_barrel_image', self._barrel_image_cb)
        self.create_service(GenerateReport, '/generate_report', self._generate_report_cb)

        self.get_logger().info('Report generator ready.')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _ring_cb(self, msg):
        self.rings = list(zip(msg.ids, msg.colors))

    def _cylinder_cb(self, msg):
        self.cylinders = list(zip(msg.ids, msg.colors, msg.orientations, msg.leaking))

    def _task_assignment_cb(self, request, response):
        self.task_assignments[request.task] = {
            'person': request.person_name,
            'cell': request.cell,
        }
        self.get_logger().info(f'Task "{request.task}" assigned by {request.person_name}')
        response.success = True
        return response

    def _anomaly_tile_cb(self, request, response):
        tile_img_path = ''
        mask_img_path = ''

        if request.tile_image.data:
            try:
                cv_img = self.bridge.imgmsg_to_cv2(request.tile_image, 'bgr8')
                tile_img_path = os.path.join(self._tmp_dir, f'tile_{request.tile_id}.jpg')
                cv2.imwrite(tile_img_path, cv_img)
            except Exception as e:
                self.get_logger().warn(f'Could not save tile image: {e}')

        if request.mask_image.data:
            try:
                cv_mask = self.bridge.imgmsg_to_cv2(request.mask_image, 'bgr8')
                mask_img_path = os.path.join(self._tmp_dir, f'mask_{request.tile_id}.jpg')
                cv2.imwrite(mask_img_path, cv_mask)
            except Exception as e:
                self.get_logger().warn(f'Could not save mask image: {e}')

        self.anomaly_tiles.append({
            'tile_id': request.tile_id,
            'anomaly_detected': request.anomaly_detected,
            'tile_img_path': tile_img_path,
            'mask_img_path': mask_img_path,
        })
        response.success = True
        return response

    def _barrel_image_cb(self, request, response):
        if not request.image.data:
            response.success = False
            return response
        try:
            cv_img = self.bridge.imgmsg_to_cv2(request.image, 'bgr8')
            img_path = os.path.join(self._tmp_dir, f'barrel_{request.barrel_id}.jpg')
            cv2.imwrite(img_path, cv_img)
            self.barrel_images[request.barrel_id] = img_path
            response.success = True
        except Exception as e:
            self.get_logger().warn(f'Could not save barrel image: {e}')
            response.success = False
        return response

    def _generate_report_cb(self, request, response):
        robot_name = request.robot_name if request.robot_name else 'R2D2'
        output_path = request.output_path if request.output_path else os.path.expanduser('~/inspection_report.pdf')

        try:
            self._build_pdf(robot_name, output_path)
            response.success = True
            response.pdf_path = output_path
            response.message = f'Report saved to {output_path}'
            self.get_logger().info(f'Report generated: {output_path}')
        except Exception as e:
            response.success = False
            response.pdf_path = ''
            response.message = str(e)
            self.get_logger().error(f'Report generation failed: {e}')

        return response

    # ------------------------------------------------------------------
    # PDF building
    # ------------------------------------------------------------------

    def _build_pdf(self, robot_name, output_path):
        parent = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(parent, exist_ok=True)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()
        s_title = ParagraphStyle('rep_title', parent=styles['Heading1'], fontSize=14, spaceAfter=4)
        s_section = ParagraphStyle(
            'rep_section', parent=styles['Heading2'], fontSize=11,
            textColor=_HEADER_COLOR, spaceAfter=4,
        )
        s_normal = styles['Normal']
        s_indent = ParagraphStyle('rep_indent', parent=s_normal, leftIndent=12, spaceAfter=2)
        s_indent2 = ParagraphStyle('rep_indent2', parent=s_normal, leftIndent=24, spaceAfter=2)

        story = []

        # ---- Header ----
        story.append(Paragraph('Inspection report', s_title))
        story.append(Paragraph(f'Date: {datetime.now().strftime("%-d.%-m.%Y")}', s_normal))
        story.append(Paragraph(f'Robot: {robot_name}', s_normal))
        story.append(Spacer(1, 0.4 * cm))
        story.append(HRFlowable(width='100%', thickness=0.5, color=colors.black))
        story.append(Spacer(1, 0.3 * cm))

        has_content = False

        # ---- Ring counting ----
        ring_task = self.task_assignments.get('rings')
        if ring_task or self.rings:
            has_content = True
            self._section_rings(story, ring_task, s_section, s_normal, s_indent, s_indent2)
            story.append(Spacer(1, 0.5 * cm))

        # ---- Barrel inspection ----
        barrel_task = self.task_assignments.get('barrels')
        if barrel_task or self.cylinders:
            has_content = True
            self._section_barrels(story, barrel_task, s_section, s_normal, s_indent)
            story.append(Spacer(1, 0.5 * cm))

        # ---- Anomaly detection ----
        anomaly_task = self.task_assignments.get('anomaly')
        if anomaly_task or self.anomaly_tiles:
            has_content = True
            self._section_anomaly(story, anomaly_task, s_section, s_normal, s_indent)

        if not has_content:
            story.append(Paragraph('No inspection data recorded.', s_normal))

        doc.build(story)

    # ------------------------------------------------------------------
    # Section helpers
    # ------------------------------------------------------------------

    def _section_rings(self, story, task_info, s_section, s_normal, s_indent, s_indent2):
        story.append(Paragraph('<u>Task: Ring Counting</u>', s_section))
        if task_info:
            story.append(Paragraph(f'Requested by: {task_info["person"]}', s_normal))
        story.append(Paragraph('Results:', s_normal))
        story.append(Paragraph(f'Total number of rings detected: {len(self.rings)}', s_indent))

        if self.rings:
            color_counts = Counter(color for _, color in self.rings)
            story.append(Paragraph('Detected colors:', s_indent))
            for clr, cnt in sorted(color_counts.items()):
                story.append(Paragraph(f'{clr.capitalize()}: {cnt}', s_indent2))

    def _section_barrels(self, story, task_info, s_section, s_normal, s_indent):
        story.append(Paragraph('<u>Task: Barrel inspection</u>', s_section))
        if task_info:
            story.append(Paragraph(f'Requested by: {task_info["person"]}', s_normal))
        story.append(Paragraph('Results:', s_normal))
        story.append(Paragraph(f'Total number of barrels inspected: {len(self.cylinders)}', s_indent))

        if not self.cylinders:
            return

        sorted_cyls = sorted(self.cylinders, key=lambda x: x[0])

        table_data = [['Barrel ID', 'Colour', 'Orientation', 'Leak detected']]
        for bid, clr, orientation, leaking in sorted_cyls:
            table_data.append([
                str(bid),
                clr.capitalize(),
                orientation.capitalize(),
                'Yes' if leaking else 'No',
            ])

        t = Table(table_data, colWidths=[2.5 * cm, 3 * cm, 4 * cm, 4 * cm])
        t.setStyle(self._base_table_style())
        for i, (bid, clr, orientation, leaking) in enumerate(sorted_cyls):
            if leaking:
                t.setStyle(TableStyle([
                    ('TEXTCOLOR', (3, i + 1), (3, i + 1), colors.red),
                    ('FONTNAME', (3, i + 1), (3, i + 1), 'Helvetica-Bold'),
                ]))
        story.append(t)

        # Images of leaking barrels
        for bid, clr, orientation, leaking in sorted_cyls:
            if leaking and bid in self.barrel_images:
                story.append(Spacer(1, 0.2 * cm))
                story.append(Paragraph(f'ID {bid}:', s_normal))
                try:
                    story.append(RLImage(self.barrel_images[bid], width=6 * cm, height=4 * cm))
                except Exception:
                    pass

    def _section_anomaly(self, story, task_info, s_section, s_normal, s_indent):
        story.append(Paragraph('<u>Task: Anomaly detection</u>', s_section))
        if task_info:
            story.append(Paragraph(f'Requested by: {task_info["person"]}', s_normal))
            cell = task_info.get('cell', '')
            if cell and cell != 'none':
                story.append(Paragraph(f'Cell: {cell.capitalize()}', s_normal))

        total = len(self.anomaly_tiles)
        ok = sum(1 for t in self.anomaly_tiles if not t['anomaly_detected'])
        nok = total - ok

        story.append(Paragraph('Results:', s_normal))
        story.append(Paragraph(f'Total number of tiles inspected: {total}', s_indent))
        story.append(Paragraph(f'OK: {ok}', s_indent))
        story.append(Paragraph(f'NOK: {nok}', s_indent))

        if not self.anomaly_tiles:
            return

        sorted_tiles = sorted(self.anomaly_tiles, key=lambda x: x['tile_id'])

        table_data = [['Tile ID', 'Status']]
        for tile in sorted_tiles:
            table_data.append([str(tile['tile_id']), 'NOK' if tile['anomaly_detected'] else 'OK'])

        t = Table(table_data, colWidths=[3 * cm, 3 * cm])
        t.setStyle(self._base_table_style())
        for i, tile in enumerate(sorted_tiles):
            clr = colors.red if tile['anomaly_detected'] else colors.green
            t.setStyle(TableStyle([
                ('TEXTCOLOR', (1, i + 1), (1, i + 1), clr),
                ('FONTNAME', (1, i + 1), (1, i + 1), 'Helvetica-Bold'),
            ]))
        story.append(t)

        # Images for NOK tiles
        for tile in sorted_tiles:
            if not tile['anomaly_detected']:
                continue
            has_tile_img = tile['tile_img_path'] and os.path.exists(tile['tile_img_path'])
            has_mask_img = tile['mask_img_path'] and os.path.exists(tile['mask_img_path'])
            if not has_tile_img and not has_mask_img:
                continue

            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph(f'ID {tile["tile_id"]}:', s_normal))

            imgs = []
            if has_tile_img:
                try:
                    imgs.append(RLImage(tile['tile_img_path'], width=5 * cm, height=4 * cm))
                except Exception:
                    pass
            if has_mask_img:
                try:
                    imgs.append(RLImage(tile['mask_img_path'], width=5 * cm, height=4 * cm))
                except Exception:
                    pass

            if len(imgs) == 2:
                img_table = Table([imgs], colWidths=[5.5 * cm, 5.5 * cm])
                img_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
                story.append(img_table)
            elif imgs:
                story.append(imgs[0])

    @staticmethod
    def _base_table_style():
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), _HEADER_COLOR),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), _ROW_COLORS),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ])


def main(args=None):
    rclpy.init(args=args)
    node = ReportGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
