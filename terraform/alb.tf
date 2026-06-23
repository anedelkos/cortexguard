resource "aws_lb" "cloud_api" {
  name               = "${local.name_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.selected.ids
}

resource "aws_lb_target_group" "cloud_api" {
  name        = "${local.name_prefix}-tg"
  port        = 8001
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.selected.id
  target_type = "ip"

  health_check {
    path                = "/healthz/live"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
    matcher             = "200"
  }
}

# HTTP listener: redirects to HTTPS when acm_certificate_arn is set, otherwise forwards directly.
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.cloud_api.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = var.acm_certificate_arn != "" ? "redirect" : "forward"
    target_group_arn = var.acm_certificate_arn == "" ? aws_lb_target_group.cloud_api.arn : null

    dynamic "redirect" {
      for_each = var.acm_certificate_arn != "" ? [1] : []
      content {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }
  }
}

# HTTPS listener: only created when acm_certificate_arn is provided.
resource "aws_lb_listener" "https" {
  count             = var.acm_certificate_arn != "" ? 1 : 0
  load_balancer_arn = aws_lb.cloud_api.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.acm_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.cloud_api.arn
  }
}
