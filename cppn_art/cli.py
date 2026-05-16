from __future__ import annotations
import argparse
import sys
import time


def _banner():
    print('\n  cppn-art  --  infinite fractal artwork from tiny neural nets\n')


def _resolve_cppn(args):
    from .network import CPPN
    from . import presets as P
    if args.preset:
        data = P.get(args.preset)
        print(f"  {data['emoji']}  Preset '{args.preset}':  {data['description']}")
        return CPPN(seed=data['seed'], hidden_sizes=data['hidden_sizes'])
    return CPPN(seed=args.seed)


def _preset_arg(p):
    g = p.add_mutually_exclusive_group()
    g.add_argument('--seed', type=int, default=42, metavar='N', help='integer seed (default: 42)')
    g.add_argument('--preset', type=str, metavar='NAME', help='use a built-in preset')


def cmd_generate(args):
    from .renderer import render_image
    _banner()
    cppn = _resolve_cppn(args)
    print(f'  {cppn.describe()}')
    print(f'  Rendering {args.width}x{args.height} image ...')
    t0 = time.time()
    img = render_image(cppn, args.width, args.height, t=args.time, scale=args.scale)
    out = args.output or f'cppn_seed{cppn.seed}.png'
    img.save(out)
    print(f'  Saved -> {out}  ({time.time()-t0:.2f}s)\n')


def cmd_animate(args):
    from .renderer import render_gif
    _banner()
    cppn = _resolve_cppn(args)
    print(f'  {cppn.describe()}')
    out = args.output or f'cppn_seed{cppn.seed}.gif'
    print(f'  Rendering {args.frames} frames at {args.width}x{args.height} ...')
    render_gif(cppn, out, width=args.width, height=args.height,
               frames=args.frames, fps=args.fps, scale=args.scale, verbose=True)
    print(f'  Saved -> {out}\n')


def cmd_gallery(args):
    from .renderer import render_gallery
    _banner()
    seeds = list(range(args.start, args.start + args.count))
    out = args.output or 'cppn_gallery.png'
    print(f'  Generating gallery: {len(seeds)} seeds, {args.cols} columns, {args.cell_size}px cells ...\n')
    render_gallery(seeds, out, cols=args.cols, cell_size=args.cell_size, verbose=True)
    print('  Done\n')


def cmd_presets(args):
    from . import presets as P
    _banner()
    print('  Built-in presets\n')
    print(f"  {'Name':<14}{'Emoji':<6}{'Seed':<10}Description")
    print('  ' + '-' * 70)
    for name, data in P.PRESETS.items():
        print(f"  {name:<14}{data['emoji']:<6}{data['seed']:<10}{data['description']}")
    print()
    print('  Usage:  cppn-art generate --preset <name>')
    print('          cppn-art animate  --preset <name>\n')


def cmd_info(args):
    _banner()
    cppn = _resolve_cppn(args)
    print(f'  {cppn.describe()}\n')
    print(f"  {'Layer':<6}  {'Shape':<18}  Activation")
    print('  ' + '-' * 45)
    in_size = 5 + cppn.z_dim
    print(f"  {'input':<6}  ({in_size},){'':12}  (x, y, r, sin_t, cos_t, z)")
    for i, layer in enumerate(cppn.layers):
        w, h = layer.W.shape
        print(f"  {i:<6}  ({w}, {h}){'':10}  {layer.activation_name}")
    ow, oh = cppn.out_layer.W.shape
    print(f"  {'out':<6}  ({ow}, {oh}){'':10}  sigmoid -> RGB")
    total = sum(l.W.size + l.b.size for l in cppn.layers) + cppn.out_layer.W.size + cppn.out_layer.b.size
    print(f'\n  Total parameters: {total:,}\n')


def main():
    parser = argparse.ArgumentParser(
        prog='cppn-art',
        description='Generate infinite fractal artwork with tiny neural networks.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=('examples:\n'
                '  cppn-art generate --seed 42\n'
                '  cppn-art generate --preset cosmos --width 1024\n'
                '  cppn-art animate  --preset aurora --frames 90 --fps 30\n'
                '  cppn-art gallery  --start 1 --count 16 --cols 4\n'
                '  cppn-art presets\n'),
    )
    sub = parser.add_subparsers(dest='command', required=True, metavar='command')

    p = sub.add_parser('generate', help='render a single PNG image')
    _preset_arg(p)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--height', type=int, default=512)
    p.add_argument('--time', type=float, default=0.0)
    p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--output', type=str)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser('animate', help='render a looping animated GIF')
    _preset_arg(p)
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--height', type=int, default=256)
    p.add_argument('--frames', type=int, default=60)
    p.add_argument('--fps', type=int, default=20)
    p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--output', type=str)
    p.set_defaults(func=cmd_animate)

    p = sub.add_parser('gallery', help='render a grid of images')
    p.add_argument('--start', type=int, default=1)
    p.add_argument('--count', type=int, default=16)
    p.add_argument('--cols', type=int, default=4)
    p.add_argument('--cell-size', type=int, default=256, dest='cell_size')
    p.add_argument('--output', type=str)
    p.set_defaults(func=cmd_gallery)

    p = sub.add_parser('presets', help='list all built-in presets')
    p.set_defaults(func=cmd_presets)

    p = sub.add_parser('info', help='show network architecture')
    _preset_arg(p)
    p.set_defaults(func=cmd_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
